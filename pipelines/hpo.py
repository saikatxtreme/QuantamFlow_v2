import os, joblib, pandas as pd, numpy as np, time
from quantumflow_core import load_config, read_csv, prepare_features, select_and_train
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform
import optuna

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    import lightgbm as lgb
    cv = 3
    n = len(y)
    fold = n // (cv+1)
    rmses = []
    for i in range(cv):
        tr = slice(0, fold*(i+1))
        va = slice(fold*(i+1), fold*(i+2))
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        from sklearn.metrics import mean_squared_error
        rmses.append(mean_squared_error(y[va], pred, squared=False))
    return float(sum(rmses)/len(rmses))

def run_optuna(X, y, n_trials=30):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials)
    return study.best_params

def main(cfg_path="configs/dev.yaml", sku_id=None):
    cfg = load_config(cfg_path)
    data_dir = cfg.get("data_dir","data")
    sales = read_csv(os.path.join(data_dir,"sales.csv"))
    if sku_id:
        sales = sales[sales['SKU_ID']==sku_id]
    feats = prepare_features(sales)
    X = feats.drop(columns=['Date','SKU_ID','Sales_Channel','Sales_Quantity']).values
    y = feats['Sales_Quantity'].values
    best = run_optuna(X, y, n_trials=20)
    out = os.path.join('artifacts', f'hpo_{sku_id or "ALL"}.json')
    os.makedirs('artifacts', exist_ok=True)
    with open(out,'w') as f:
        import json
        json.dump(best, f)
    print('Wrote best params to', out)

if __name__ == '__main__':
    main()
