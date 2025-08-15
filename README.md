# Quantumflow – Demand Forecasting & Inventory Planning (API-first)

This is a production-ready scaffold that turns your Streamlit POC into an API + library you can deploy on **Google Cloud Run (free tier)** or similar.

## What you get

- **Core library** (`quantumflow_core/`) with:
  - Data schemas (Pydantic)
  - Feature engineering (calendar, lags, rolling stats)
  - Time-series **blocked CV**
  - **LightGBM** regressors (+ quantile models) and **XGBoost** fallback
  - Forecast pipeline (train → select best → predict)
  - Inventory policy & indent calculation (MOQ, multiples, shelf-life, service levels)
- **FastAPI** service (`apps/api/`) with endpoints:
  - `POST /train` – trains models from CSVs in `data/` or GCS
  - `POST /forecast` – returns forecasts for SKUs
  - `POST /indent` – returns SKU/component order recommendations
  - `GET /health` – health check
- **Dockerfile** and **Cloud Run** deploy workflow
- **Configs** (`configs/`) for dev/prod

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn apps.api.main:app --reload --port 8080
```

Open http://localhost:8080/docs for Swagger.

## Deploy to Google Cloud Run (free tier)

1. Build image:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/quantumflow:latest
```
2. Deploy:
```bash
gcloud run deploy quantumflow --image gcr.io/PROJECT_ID/quantumflow:latest --platform managed --region asia-south1 --allow-unauthenticated --memory 1Gi --cpu 1
```

## Data

Place your CSVs in `data/` (or configure GCS URIs in `configs/prod.yaml`). Expected tables:
- `sales.csv` – Date, SKU_ID, Sales_Channel, Sales_Quantity
- `inventory.csv` – Date, SKU_ID, On_Hand
- `promos.csv` – Date, SKU_ID, Promo_Flag
- `external.csv` – Date, var...
- `leadtime.csv` – SKU_ID, Lead_Time_Days
- `bom.csv` – Parent_SKU, Component_SKU, Qty_Per
- `global_config.yaml` – holding cost etc.

## Notes
- LightGBM provides **quantile** forecasts used to align to **service levels**.
- We use **blocked CV** to choose best model per SKU×Channel.
- For performance, training fans out by SKU×Channel; scale with threads or Ray later.


## MLflow Tracking
By default uses a local `./mlruns` directory. To use a server, set:
```
export MLFLOW_TRACKING_URI=http://your-mlflow:5000
```

## Prefect (scheduled runs)
Run the nightly flow locally:
```
python -m pipelines.flow
```
Or register on Prefect Cloud and create a schedule.

## GitHub Actions → Cloud Run
Set these repository secrets:
- `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_SA_EMAIL`, `GCP_WORKLOAD_IDP`

Push to `main` to build and deploy automatically.


## Additional pipelines & tools included
- `pipelines/backfill_weather.py` - backfill weather cache and produce `data/sales_enriched.parquet`
- `pipelines/hpo.py` - hyperparameter tuning (Optuna) per SKU or all SKUs
- `pipelines/drift_monitor.py` - uses Evidently to generate a drift report and notify Slack (requires Slack token)
- `dashboard/` - React dashboard scaffold (minimal instructions)

## Example workflow to perform full production readiness
1. Backfill weather cache for all locations:
   ```bash
   python pipelines/backfill_weather.py --cfg configs/prod.yaml
   ```
2. Run hyperparameter tuning for a top SKU:
   ```bash
   python pipelines/hpo.py --cfg configs/prod.yaml --sku_id A1
   ```
3. Train with tuned params (update config or artifacts)
   ```bash
   python pipelines/train.py --cfg configs/prod.yaml
   ```
4. Run the drift monitor weekly and send report to Slack:
   ```bash
   export SLACK_TOKEN=xoxb-...
   python pipelines/drift_monitor.py
   ```
