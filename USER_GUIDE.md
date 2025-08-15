# Quantumflow - User Guide (Functional Document)

## Overview
Quantumflow is a demand forecasting, inventory planning, and indent recommendation system. It provides APIs for forecasting and indent calculation and a batch training pipeline with MLflow tracking and external factors enrichment (weather & holidays).

## Key Features
- Time-series forecasting using LightGBM / XGBoost with blocked CV
- Quantile forecasting for service-level-driven reorder points
- External enrichment: weather (Open-Meteo) and holiday flags (python-holidays)
- Per-SKU location mapping for regional weather enrichment
- Inventory indent logic: MOQ, multiples, shelf-life caps, reorder point calculation
- MLflow tracking of training runs and feature importances
- Prefect flow for scheduled training

## Files of Interest
- `quantumflow_core/` : core library
  - `external_factors.py` : weather and holiday enrichment + batch_enrich_weather
  - `features.py` : feature engineering; calls enrichment if configured
  - `models.py` : model training utilities
  - `inventory.py` : indent recommendation logic
- `pipelines/train.py` : training pipeline (logs to MLflow by default)
- `apps/api/main.py` : FastAPI app exposing `/train`, `/load`, `/forecast`, `/indent`
- `data/sku_locations.csv` : sample SKU -> lat/lon mapping for batch weather enrichment
- `configs/` : dev/prod configs

## Quickstart (local)
1. Prepare data folder with CSVs:
   - `sales.csv` : Date, SKU_ID, Sales_Channel, Sales_Quantity, optional lat/lon
   - `sku_locations.csv` : SKU_ID, lat, lon  (useful for regional weather)
   - `promos.csv`, `bom.csv`, `leadtime.csv` as needed
2. Install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Train locally and populate MLflow:
   ```bash
   python pipelines/train.py
   ```
4. Start API:
   ```bash
   uvicorn apps.api.main:app --reload --port 8080
   ```
5. Load model to API and request forecasts:
   ```bash
   curl -X POST http://localhost:8080/load
   curl -X POST http://localhost:8080/forecast -H "Content-Type: application/json" -d '{"rows":[...]}'
   ```

## How to use per-SKU weather enrichment
- Provide `data/sku_locations.csv` mapping SKU_ID to lat/lon.
- `pipelines/train.py` will detect it and batch-fetch weather into `data/weather_cache/` and merge weather columns into training data.
- The API `forecast` expects historical rows sufficient to compute lags or you can serve pre-computed features.

## Best practices
- Backfill weather cache for all SKU locations before training to avoid API latency
- Use `MLFLOW_TRACKING_URI` to point to a shared MLflow server when working in a team
- Tune LightGBM parameters via parameter search using blocked CV for each SKU group
- Keep `sku_locations.csv` up to date as SKUs map to different Fulfillment Centers/Regions

## Troubleshooting
- If weather enrichment fails due to API issues, training falls back to original data and logs warnings
- Ensure the `Date` column is ISO-format (`YYYY-MM-DD`) for correct merges
- If MLflow artifact logging fails, check write permissions for `./mlruns` or configured tracking URI
