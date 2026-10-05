# Setting up the shared MLflow store

By default an evaluation run is logged to a local `./mlruns` folder (thrown away on a CI runner). The shared store makes a run from your
laptop and the weekly CI run land in one place: **MLflow's tables in your Neon Postgres** (in the `public` schema, apart from the
`rag_*` schemas) and **the run's files in a bucket folder**. The code is ready; these are the steps only you can do.

## 1. What turns it on (no secrets involved)

| Where | Setting | Value |
|---|---|---|
| your laptop (`.env`) | `MLFLOW_STORE` | `database` (it then uses your existing `DATABASE_URL`) |
| your laptop (`.env`) | `MLFLOW_ARTIFACT_ROOT` | `gs://crossscan-docs-upload-here/mlflow` |
| GitHub repository **variables** (Settings, Secrets and variables, Actions, Variables) | `MLFLOW_STORE`, `MLFLOW_ARTIFACT_ROOT` | the same two values |

Leave them unset and nothing changes (local `./mlruns`, and the weekly CI run does not log).

## 2. The Python driver (once, on your laptop)

MLflow talks to Postgres through SQLAlchemy and needs `psycopg2`. It is in `requirements.txt` (`psycopg2-binary`), so
`pip install -r requirements.txt` in your virtual environment installs it.

## 3. Create MLflow's tables (once)

From the project root, with `MLFLOW_STORE=database` in your `.env`:

```bash
PYTHONPATH=src python -c "from evaluation import experiment_tracking as et; import mlflow; mlflow.set_tracking_uri(et.tracking_uri()); mlflow.set_experiment(et.EXPERIMENT_NAME); print('MLflow store ready:', et.tracking_uri().split('@')[-1])"
```

MLflow creates about 20 tables (names such as `experiments`, `runs`, `metrics`) in the `public` schema. Nothing in `rag_new` or `rag_v2` is touched.
To see them: `mlflow ui --backend-store-uri "<the same URI>"` (the URI is `DATABASE_URL` with `postgresql://` changed to `postgresql+psycopg2://`).

## 4. Let the weekly CI run write the files (once)

The weekly workflow signs in to Google as a dedicated account, `eval-runner`, that may only write to the bucket. Run as the project owner:

```bash
PROJECT_ID=sagar-crossscan
PROJECT_NUMBER=807612796446
REPO=Ifsaurabh/crossscan_multimodal_agentic_rag
EVAL=eval-runner@$PROJECT_ID.iam.gserviceaccount.com

gcloud iam service-accounts create eval-runner --display-name="Weekly evaluation (writes MLflow files)" --project $PROJECT_ID

# the evaluation may write and read objects in the bucket (its `mlflow/` folder is where the files go)
gcloud storage buckets add-iam-policy-binding gs://crossscan-docs-upload-here \
  --member="serviceAccount:$EVAL" --role=roles/storage.objectUser

# GitHub's keyless identity (already set up for the deploys) may act as this account
gcloud iam service-accounts add-iam-policy-binding $EVAL --project $PROJECT_ID \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/$REPO"
```

The trust in `github-pool` is limited to the `main` branch, which is where the scheduled run executes.
Your laptop signs in with your own Google login (`gcloud auth application-default login`), which already has bucket access.

## 5. Check

Run a small evaluation from your laptop, then open the MLflow UI and look for the experiment `crossscan-rag-eval`:

```bash
PYTHONPATH=src python -m evaluation.run_evaluation --only-extra --golden-extra data/golden_set_m3.jsonl --limit 3
```

(It costs a few Gemini calls: 3 questions, no judge metrics.)
