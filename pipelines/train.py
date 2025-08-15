import pandas as pd, numpy as np, os, joblib, mlflow, time
from pathlib import Path
from quantumflow_core import load_config, read_csv, ensure_columns, prepare_features, select_and_train
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","file:./mlruns"))
mlflow.set_experiment("quantumflow_forecasting")

def main(cfg_path="configs/dev.yaml"):
    cfg = load_config(cfg_path)
    data_dir = cfg.get("data_dir","data")
    source = cfg.get("data_source","local")
    sales_path = os.path.join(data_dir,"sales.csv") if source=="local" else f"gs://{cfg['gcs_bucket']}/{cfg['gcs_prefix']}/sales.csv"
    sales = read_csv(sales_path)
    ensure_columns(sales, ["Date","SKU_ID","Sales_Channel","Sales_Quantity"], "sales")
    with mlflow.start_run(run_name=f"train_{int(time.time())}"):
        from quantumflow_core.external_factors import batch_enrich_weather
sku_map = os.path.join(data_dir, 'sku_locations.csv')
if os.path.exists(sku_map):
    print('Found SKU location map, running batch weather enrichment...')
    sales = batch_enrich_weather(sales, sku_location_map_path=sku_map, cache_dir=os.path.join(data_dir,'weather_cache'))
feats = prepare_features(sales, enrich_weather=False)

        model = select_and_train(feats)
        mlflow.log_param("selected_model", model.name)
        mlflow.log_param("features", ",".join(model.features))
        out = Path("artifacts"); out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out/"model.joblib")
        mlflow.log_artifact(str(out/"model.joblib"))
        # Log feature importances if available
        try:
            import pandas as _pd
            fi = None
            if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                # TrainedModel wrapper case
                fi = model.model.feature_importances_
                names = model.features
            elif hasattr(model, 'feature_importances_'):
                fi = model.feature_importances_
                names = model.features if 'model' not in locals() else model.features
            if fi is not None:
                fi_df = _pd.DataFrame({'feature': names, 'importance': fi})
                fi_path = out / 'feature_importances.csv'
                fi_df.to_csv(fi_path, index=False)
                mlflow.log_artifact(str(fi_path))
        except Exception:
            pass
        
        print("Saved model to artifacts/model.joblib")

if __name__ == "__main__":
    main()
