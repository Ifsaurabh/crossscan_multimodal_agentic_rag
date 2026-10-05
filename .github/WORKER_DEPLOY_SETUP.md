# Deploying the ingestion worker from GitHub Actions

The worker is deployed by the **same pipeline as the app** (`.github/workflows/tests.yml`, job `deploy-worker`). Nothing is
built or deployed by hand.

```
push to main -> unit + integration tests -> (only if they pass, and only if a worker file changed)
             -> build Dockerfile.worker (the build runs ingestion.image_check: every model and tool must load)
             -> push to Artifact Registry
             -> create or update the Cloud Run JOB `crossscan-ingest-worker` with the new image
```

"A worker file changed" means `Dockerfile.worker`, `Dockerfile.worker.dockerignore`, `requirements-worker.txt`, `src/ingestion/` or
`src/shared/`. A push that touches none of them deploys nothing. **Actions -> Tests and deploy -> Run workflow** deploys on demand.

The deploy only creates or updates the job. **It never starts a run**, so a deploy never touches the queue. A run starts when you
execute the job (Cloud Run -> Jobs -> `crossscan-ingest-worker` -> Execute), or later when a scheduler does. A run takes everything
queued, one document at a time, and ends after two empty pulls.

Pull requests that touch the worker only build the image and check it (`.github/workflows/worker-image-check.yml`); they deploy nothing.

## One-time Google Cloud setup

The workflow signs in with the keyless identity already set up for the app (`DEPLOY_SETUP.md`: `github-deployer`, `github-pool`).
Only three permissions are new. Run them once, as the project owner (or do the same in the console, IAM page):

```bash
PROJECT_ID=sagar-crossscan
DEPLOYER=github-deployer@$PROJECT_ID.iam.gserviceaccount.com
WORKER=ingest-worker@$PROJECT_ID.iam.gserviceaccount.com

# 1. The deployer may create and update Cloud Run jobs. Creating a job needs the role on the project
#    (the app's role is on the one service only). It covers every Cloud Run service and job in the project.
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$DEPLOYER" --role=roles/run.developer

# 2. The deployer may run a job AS the worker account
gcloud iam service-accounts add-iam-policy-binding $WORKER --project $PROJECT_ID \
  --member="serviceAccount:$DEPLOYER" --role=roles/iam.serviceAccountUser

# 3. The worker may read the database password (the same secret the app uses)
gcloud secrets add-iam-policy-binding DATABASE_URL --project $PROJECT_ID \
  --member="serviceAccount:$WORKER" --role=roles/secretmanager.secretAccessor
```

Already in place, nothing to do: the worker's access to the bucket (Storage Object Admin) and to the queue (Pub/Sub Subscriber on
`doc-events-sub`); the deployer's right to push images; the `HF_TOKEN` repository secret (Prompt Guard, as for the app).

The Artifact Registry repository (`cloud-run-source-deploy`) is shared with the app; the worker image is
`.../crossscan-ingest-worker`, next to the app's.

## What the job runs with

Set by the workflow on every deploy (edit them in `tests.yml`, job `deploy-worker`, `env:` and the `gcloud run jobs deploy` step):

| Setting | Value | Why |
|---|---|---|
| `SCHEMA_NAME` | `rag_v2` | the new tables; the app keeps `rag_new` until the switch |
| `DOCS_BUCKET`, `DOCS_SUBSCRIPTION` | `crossscan-docs-upload-here`, `doc-events-sub` | where documents arrive |
| `DATABASE_URL` | secret `DATABASE_URL` | the same Neon database |
| CPU, memory | 4 vCPU, 8 GiB | a starting guess for Docling and the models: **tune it after the first real run** |
| `--tasks 1`, `--max-retries 0` | one task, no automatic retry | a failure must show in the log and the dead-letter queue, not loop and burn compute |
| `--task-timeout` | 3600 s | one hour per run |

The worker needs no Gemini key (it makes no model calls).

## First time

1. Do the three permissions above.
2. Merge the change to `main` (or run the workflow by hand). The first image build is the slow one (expect 30 to 40 minutes);
   later builds reuse the cached layers from the registry.
3. When `deploy-worker` is green, the job exists. Upload one document into `incoming/<domain>/` and **Execute** the job; watch the
   log: every stage prints one line with its seconds and tokens.
4. `PYTHONPATH=src SCHEMA_NAME=rag_v2 python -m ingestion.ingestion_reports --recent 5` (or `--summary`) shows the saved reports.

## Known gaps

- **Overlap:** handled by the lease row (`worker_lease`): an execution that does not get the lease exits at once. The table is created by `python -m ingestion.pipeline_schema --setup` (done in `rag_v2`).
- **Trigger:** one execution per upload, started by the function described under "The upload trigger" below (you can also Execute the job by hand).
- **Many uploads at once:** one execution gets the lease and handles the queue one document at a time; the others exit within seconds. A 12-paper batch takes roughly an hour or more, so keep `--task-timeout` generous (the workflow sets 3600 s; raise it if batches grow).
- **Not tested on a real build yet:** no Docker was available while this was written. The first pull request or deploy is the real test.

## The upload trigger (once)

`functions/ingest_trigger/` is a small Cloud Run function. It is deployed by the same pipeline (job `deploy-trigger` in `tests.yml`, only when its
own files change) and gives the worker one execution per upload. It listens to the topic `doc-events` through a subscription of its own, so the worker's
subscription `doc-events-sub` is not affected. Run once, as the project owner, **after the worker job exists** (after the first `deploy-worker`):

```bash
PROJECT_ID=sagar-crossscan
REGION=europe-west1
DEPLOYER=github-deployer@$PROJECT_ID.iam.gserviceaccount.com
TRIGGER=ingest-trigger@$PROJECT_ID.iam.gserviceaccount.com
PROJECT_NUMBER=807612796446

# 1. APIs the function needs
gcloud services enable cloudfunctions.googleapis.com eventarc.googleapis.com cloudbuild.googleapis.com run.googleapis.com pubsub.googleapis.com --project $PROJECT_ID

# 2. The function's own account: it may receive events and run the worker job, nothing else
gcloud iam service-accounts create ingest-trigger --display-name="Starts the ingestion worker on an upload" --project $PROJECT_ID
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$TRIGGER" --role=roles/eventarc.eventReceiver
gcloud run jobs add-iam-policy-binding crossscan-ingest-worker --region $REGION --project $PROJECT_ID \
  --member="serviceAccount:$TRIGGER" --role=roles/run.invoker
# if the first upload then fails with "permission denied", the invoker role does not cover starting a job in your project:
# grant roles/run.developer on the job instead (same command, other role)

# 3. The deployer may deploy functions, and run them as that account
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$DEPLOYER" --role=roles/cloudfunctions.developer
gcloud iam service-accounts add-iam-policy-binding $TRIGGER --project $PROJECT_ID \
  --member="serviceAccount:$DEPLOYER" --role=roles/iam.serviceAccountUser
gcloud iam service-accounts add-iam-policy-binding $PROJECT_NUMBER-compute@developer.gserviceaccount.com --project $PROJECT_ID \
  --member="serviceAccount:$DEPLOYER" --role=roles/iam.serviceAccountUser
```

Then push (or run the workflow by hand): `deploy-trigger` deploys the function. Check: upload a file into `incoming/<domain>/`; in Cloud Run, Jobs, `crossscan-ingest-worker`,
a new execution appears within seconds, and the function's log says `started ... for incoming/...`. Several uploads at once start several executions; one works, the others log
"Another worker run holds the lease" and exit.
