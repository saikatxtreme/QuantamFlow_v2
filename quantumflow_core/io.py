import os
import pandas as pd
from typing import Optional
try:
    from google.cloud import storage
    HAS_GCS = True
except Exception:
    HAS_GCS = False

def read_csv(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        if not HAS_GCS:
            raise RuntimeError("google-cloud-storage not installed for GCS paths")
        bucket, *prefix = path.replace("gs://","",1).split("/",1)
        prefix = prefix[0] if prefix else ""
        client = storage.Client()
        blob = client.bucket(bucket).blob(prefix)
        data = blob.download_as_bytes()
        from io import BytesIO
        return pd.read_csv(BytesIO(data))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def ensure_columns(df, cols, name=""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    return df

def write_parquet(df, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
