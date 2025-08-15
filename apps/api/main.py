from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import joblib, os, subprocess

from quantumflow_core import prepare_features
from quantumflow_core.models import TrainedModel, predict
from quantumflow_core.inventory import IndentPolicy, recommend_order

app = FastAPI(title="Quantumflow API", version="1.1.0")

MODEL_PATH = os.environ.get("QF_MODEL_PATH", "artifacts/model.joblib")
_model: Optional[TrainedModel] = None

class ForecastRequest(BaseModel):
    rows: List[Dict]
    quantile: Optional[float] = None

class IndentRequest(BaseModel):
    daily_mean_demand: float = Field(ge=0)
    daily_std_demand: float = Field(ge=0)
    lead_time_days: int = Field(gt=0)
    on_hand: float = Field(ge=0)
    service_level: float = Field(default=0.9, gt=0, le=0.999)
    moq: int = 1
    multiple: int = 1
    shelf_life_days: Optional[int] = None

@app.get("/health")
def health():
    return {"status":"ok", "model_loaded": _model is not None}

@app.post("/load")
def load_model():
    global _model
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(404, f"Model not found at {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)
    return {"loaded": True, "model": getattr(_model, "name", "unknown")}

@app.post("/train")
def train_model():
    try:
        subprocess.run(["python","pipelines/train.py"], check=True)
        return {"trained": True}
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Training failed: {e}")

@app.post("/forecast")
def forecast(req: ForecastRequest):
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, "Model not loaded. POST /load first or train a model.")
        _model = joblib.load(MODEL_PATH)
    df = pd.DataFrame(req.rows)
    feats = prepare_features(df)
    preds = predict(_model, feats, quantile=req.quantile)
    out = feats[["Date","SKU_ID","Sales_Channel"]].copy()
    out["forecast"] = preds
    return {"rows": out.to_dict(orient="records")}

@app.post("/indent")
def indent(req: IndentRequest):
    policy = IndentPolicy(service_level=req.service_level, moq=req.moq, multiple=req.multiple, shelf_life_days=req.shelf_life_days)
    rec = recommend_order(req.daily_mean_demand, req.daily_std_demand, req.lead_time_days, req.on_hand, policy)
    return rec
