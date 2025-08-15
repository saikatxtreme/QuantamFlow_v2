# Deployment Steps - Quantumflow (Production)

## 1. Prerequisites
- GCP project with Cloud Run, Cloud Build enabled
- Artifact Registry or Container Registry enabled
- Service account for CI/CD (roles: run.admin, storage.admin, cloudbuild.builds.editor, ml.admin if using MLflow server)
- GitHub repo with code and secrets configured (see GitHub Actions)

## 2. Configure environment variables (Cloud Run)
Set environment variables on Cloud Run or in your container runner:
- `QF_CONFIG` = `configs/prod.yaml`
- `MLFLOW_TRACKING_URI` = `http://<mlflow-server>/` (optional)
- `QF_MODEL_PATH` = path to model artifact (optional)

## 3. GCS & Data
- Upload your CSVs to the GCS bucket specified in `configs/prod.yaml` (sales.csv, sku_locations.csv, promos.csv, bom.csv, leadtime.csv)
- Ensure service account has access to the bucket

## 4. CI/CD (GitHub Actions)
- Configure repo secrets: `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_SA_EMAIL`, `GCP_WORKLOAD_IDP`
- Push to `main` to trigger build & deploy workflow

## 5. Run initial backfill & train
- Connect to Cloud Run container or run locally with `QF_CONFIG=configs/prod.yaml`
- Run: `python pipelines/train.py` (or POST `/train` to container)
- Verify MLflow run logged and model artifact present

## 6. Schedule nightly flows
- Use Prefect Cloud or GitHub Actions scheduled workflow to run `pipelines/flow.py` nightly
- Prefect will execute training and publish models to artifacts

## 7. Monitoring & Observability
- Configure MLflow server for central tracking (optional)
- Export LightGBM feature importances for explainability (logged as artifact)
- Integrate logs with Stackdriver / Cloud Logging

