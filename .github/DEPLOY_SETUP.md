# Deploying CrossScan from GitHub Actions

How it works (workflow: `.github/workflows/tests.yml`):

```
push to main -> unit + integration tests -> (only if they pass) build the image
             -> push to Artifact Registry -> new Cloud Run revision with NO traffic
             -> health check on that revision -> move traffic to it
```

If the tests fail, nothing is built. If the new revision fails its health check, traffic never moves and
users stay on the previous revision. The service's existing environment variables and secrets are kept
(only the image and two labels change).

Pull requests only run the tests, plus `docker-build.yml` (builds and starts the image) when the
Dockerfile, requirements or model config change. They never deploy.

## One-time Google Cloud setup (about 10 minutes)

The workflow signs in to Google Cloud **without any stored key**, using Workload Identity Federation:
Google trusts GitHub's signed identity for this one repository and only for the `main` branch.
Run these once, in a terminal where `gcloud` is logged in as the project owner.

```bash
PROJECT_ID=sagar-crossscan
PROJECT_NUMBER=807612796446
REGION=europe-west1
SERVICE=crossscan-multimodal-agentic-rag-git
REPO=Ifsaurabh/crossscan_multimodal_agentic_rag
DEPLOYER=github-deployer@$PROJECT_ID.iam.gserviceaccount.com
RUNTIME_SA=$PROJECT_NUMBER-compute@developer.gserviceaccount.com   # the account the service runs as

# 1. APIs used by the keyless sign-in
gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project $PROJECT_ID

# 2. The account the workflow acts as
gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer" --project $PROJECT_ID

# 3. Trust GitHub's identity, for this repo and the main branch only
gcloud iam workload-identity-pools create github-pool --location=global --display-name="GitHub Actions" --project $PROJECT_ID
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=github-pool --project $PROJECT_ID \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository=='$REPO' && assertion.ref=='refs/heads/main'"
gcloud iam service-accounts add-iam-policy-binding $DEPLOYER --project $PROJECT_ID \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/$REPO"

# 4. What the deployer may do: push images, update THIS one service, and run it as the runtime account
gcloud artifacts repositories add-iam-policy-binding cloud-run-source-deploy --location=$REGION --project $PROJECT_ID \
  --member="serviceAccount:$DEPLOYER" --role=roles/artifactregistry.writer
gcloud run services add-iam-policy-binding $SERVICE --region=$REGION --project $PROJECT_ID \
  --member="serviceAccount:$DEPLOYER" --role=roles/run.developer
gcloud iam service-accounts add-iam-policy-binding $RUNTIME_SA --project $PROJECT_ID \
  --member="serviceAccount:$DEPLOYER" --role=roles/iam.serviceAccountUser
```

The names above (`github-pool`, `github-provider`, `github-deployer`) are the ones the workflow expects.
If you choose different names, change `WORKLOAD_IDENTITY_PROVIDER` and `DEPLOY_SERVICE_ACCOUNT` at the top of
the `deploy` job in `tests.yml`.

## Switching over from the old Cloud Build trigger

1. Google Cloud console -> Cloud Build -> Triggers (region `europe-west1`) ->
   `cloudrun-crossscan-multimodal-agentic-rag-git-europe-west1-Imqa` -> **Disable**. Do not delete it yet:
   it is your fallback. If both stay enabled, every push would deploy twice.
2. GitHub -> Actions -> "Tests and deploy (push / PR)" -> **Run workflow** on `main`. Watch the first run;
   the first image build is the slow one (roughly 20-30 minutes), later builds reuse the cached layers.
3. Optional manual approval before each deploy: GitHub -> Settings -> Environments -> `production` ->
   Required reviewers.

## Secrets on the service (do once, after the first successful deploy)

- `NEO4J_USER` is a plain environment variable on the service. Leave it as it is: attaching a **secret**
  with the same name is refused by Cloud Run.
- The admin account and the daily limit come from Secret Manager (secrets already created):

```bash
gcloud run services update $SERVICE --region=$REGION --project $PROJECT_ID \
  --update-secrets=ADMIN_USERNAME=ADMIN_USERNAME:latest,ADMIN_PASSWORD=ADMIN_PASSWORD:latest,USER_DAILY_REQUEST_LIMIT=USER_DAILY_REQUEST_LIMIT:latest
```

(The runtime account needs the *Secret Manager Secret Accessor* role on those three secrets.)

## Rolling back

Cloud Run console -> the service -> **Revisions** -> pick the previous revision -> **Manage traffic** -> 100%.
The workflow keeps the previous revision, so this takes seconds.
