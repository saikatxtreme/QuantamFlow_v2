import os
import pandas as pd
from quantumflow_core.external_factors import batch_enrich_weather
from quantumflow_core.config import load_config

def main(cfg_path="configs/dev.yaml"):
    cfg = load_config(cfg_path)
    data_dir = cfg.get("data_dir","data")
    sku_map = os.path.join(data_dir, "sku_locations.csv")
    sales_path = os.path.join(data_dir, "sales.csv")
    if not os.path.exists(sales_path):
        raise FileNotFoundError(sales_path)
    sales = pd.read_csv(sales_path)
    print("Starting batch weather backfill...")
    enriched = batch_enrich_weather(sales, sku_location_map_path=sku_map, cache_dir=os.path.join(data_dir,"weather_cache"))
    out_path = os.path.join(data_dir, "sales_enriched.parquet")
    enriched.to_parquet(out_path, index=False)
    print(f"Wrote enriched sales to {out_path}")

if __name__ == "__main__":
    main()
