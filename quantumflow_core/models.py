from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from .evaluation import blocked_cv_slices, rmse
import warnings

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

FEATURES_BASE = [
  "Promo_Flag","dayofweek","weekofyear","month","quarter",
  "lag_1","lag_7","lag_14","roll_mean_7","roll_std_7","roll_mean_28","roll_std_28"
]

@dataclass
class ModelSpec:
    name: str
    params: dict

@dataclass
class TrainedModel:
    name: str
    model: object
    features: List[str]
    quantile_models: Dict[float, object] | None = None

def _fit_lgbm(X, y, params):
    if not HAS_LGB:
        raise RuntimeError("LightGBM not installed in environment")
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model

def _fit_xgb(X, y, params):
    if not HAS_XGB:
        raise RuntimeError("XGBoost not installed in environment")
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model

def select_and_train(df: pd.DataFrame, target="Sales_Quantity", n_splits=3) -> TrainedModel:
    X = df[FEATURES_BASE].values
    y = df[target].values
    # Candidate specs
    specs = []
    if HAS_LGB:
        specs.append(ModelSpec("lgbm", dict(n_estimators=400, learning_rate=0.05, max_depth=-1, subsample=0.9, colsample_bytree=0.8)))
    if HAS_XGB:
        specs.append(ModelSpec("xgb", dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8, tree_method="hist")))
    if not specs:
        raise RuntimeError("No GBM backend available (LightGBM or XGBoost).")

    best = None
    best_rmse = 1e18

    for spec in specs:
        # lightweight blocked CV
        rmses = []
        for tr, va in blocked_cv_slices(len(y), n_splits=n_splits):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]
            if spec.name == "lgbm":
                m = _fit_lgbm(Xtr, ytr, spec.params)
            else:
                m = _fit_xgb(Xtr, ytr, spec.params)
            pred = m.predict(Xva)
            rmses.append(rmse(yva, pred))
        cv_rmse = float(np.mean(rmses))
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best = spec

    # Train on full data
    if best.name == "lgbm":
        base_model = _fit_lgbm(X, y, best.params)
    else:
        base_model = _fit_xgb(X, y, best.params)

    # Quantile models (LightGBM only)
    quantiles = [0.5, 0.8, 0.9, 0.95]
    q_models = {}
    if best.name == "lgbm":
        for q in quantiles:
            q_params = dict(best.params)
            q_params.update(objective="quantile", alpha=q)
            q_models[q] = _fit_lgbm(X, y, q_params)

    return TrainedModel(name=best.name, model=base_model, features=list(FEATURES_BASE), quantile_models=q_models or None)

def predict(trained: TrainedModel, df_future: pd.DataFrame, quantile: float | None = None) -> np.ndarray:
    X = df_future[trained.features].values
    if quantile is not None and trained.quantile_models and quantile in trained.quantile_models:
        return trained.quantile_models[quantile].predict(X)
    return trained.model.predict(X)
