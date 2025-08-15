import pandas as pd
import numpy as np

CAL_PARTS = ["dayofweek","weekofyear","month","quarter"]

def add_calendar(df, date_col="Date"):
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["dayofweek"] = d.dt.dayofweek
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    return df

def add_lags_rollups(df, key_cols, target_col="Sales_Quantity", lags=(1,7,14), rolls=(7,28)):
    df = df.sort_values(["Date"])
    df = df.copy()
    g = df.groupby(key_cols, group_keys=False)
    for L in lags:
        df[f"lag_{L}"] = g[target_col].shift(L)
    for W in rolls:
        df[f"roll_mean_{W}"] = g[target_col].shift(1).rolling(W, min_periods=max(2, W//2)).mean()
        df[f"roll_std_{W}"]  = g[target_col].shift(1).rolling(W, min_periods=max(2, W//2)).std()
    return df


def prepare_features(sales: pd.DataFrame, promos=None, external=None, enrich_weather: bool=False, lat: float=None, lon: float=None, weather_cache_path: str=None, country_code: str='US'):
    df = sales.copy()
    if promos is not None and len(promos):
        df = df.merge(promos, on=["Date","SKU_ID"], how="left")
        df["Promo_Flag"] = df["Promo_Flag"].fillna(0).astype(int)
    else:
        df["Promo_Flag"] = 0
    if external is not None and len(external):
        df = df.merge(external, on=["Date"], how="left")
    # External enrichments (weather, holidays)
    if enrich_weather and lat is not None and lon is not None:
        try:
            from .external_factors import add_weather_features
            df = add_weather_features(df, lat, lon, cache_path=weather_cache_path, date_col='Date')
        except Exception:
            pass
    # holidays
    try:
        from .external_factors import add_holiday_flags
        df = add_holiday_flags(df, country_code=country_code, date_col='Date')
    except Exception:
        pass
    df = add_calendar(df, "Date")
    df = add_lags_rollups(df, ["SKU_ID","Sales_Channel"], "Sales_Quantity")
    df = df.dropna().reset_index(drop=True)
    return df
