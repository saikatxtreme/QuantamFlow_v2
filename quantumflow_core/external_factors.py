import os
import pandas as pd
import requests
from datetime import datetime
import holidays as hols
from cachetools import cached, TTLCache

# cache for in-memory short-term calls (not persistent)
mem_cache = TTLCache(maxsize=1024, ttl=3600)

def date_range_for_df(df, date_col="Date"):
    d = pd.to_datetime(df[date_col])
    return d.min().strftime("%Y-%m-%d"), d.max().strftime("%Y-%m-%d")

def _open_meteo_fetch(lat, lon, start_date, end_date, timezone="auto"):
    # Use Open-Meteo historical archive API
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = dict(latitude=lat, longitude=lon, start_date=start_date, end_date=end_date,
                  daily="temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                  timezone=timezone)
    resp = requests.get(base, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def parse_open_meteo(json_payload):
    # returns DataFrame with 'time', 'temp_max', 'temp_min', 'precip_mm', 'weathercode'
    daily = json_payload.get("daily", {})
    df = pd.DataFrame({
        "Date": pd.to_datetime(daily.get("time", [])),
        "temp_max": daily.get("temperature_2m_max", []),
        "temp_min": daily.get("temperature_2m_min", []),
        "precip_mm": daily.get("precipitation_sum", []),
        "weathercode": daily.get("weathercode", [])
    })
    return df

def fetch_weather_for_location(lat, lon, start_date, end_date, cache_path=None, timezone="auto"):
    # cache_path: optional local parquet path to persist fetched weather
    if cache_path and os.path.exists(cache_path):
        try:
            w = pd.read_parquet(cache_path)
            # If cache covers requested range, return subset
            wmin = w['Date'].min().date().strftime("%Y-%m-%d")
            wmax = w['Date'].max().date().strftime("%Y-%m-%d")
            if wmin <= start_date and wmax >= end_date:
                return w[(w['Date'] >= start_date) & (w['Date'] <= end_date)].reset_index(drop=True)
        except Exception:
            pass
    payload = _open_meteo_fetch(lat, lon, start_date, end_date, timezone=timezone)
    df = parse_open_meteo(payload)
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
    return df

def add_weather_features(df, lat, lon, cache_path=None, date_col="Date"):
    # df: expects Date column parseable; returns df merged with weather columns
    if df is None or len(df)==0:
        return df
    start, end = date_range_for_df(df, date_col=date_col)
    try:
        w = fetch_weather_for_location(lat, lon, start, end, cache_path=cache_path)
    except Exception as e:
        # best-effort: attach NaNs if API fails
        w = pd.DataFrame({ 'Date': pd.to_datetime(pd.Series(pd.to_datetime(df[date_col]).dt.date.unique())), 'temp_max': None, 'temp_min': None, 'precip_mm': None, 'weathercode': None })
    # normalize Date types
    w['Date'] = pd.to_datetime(w['Date']).dt.date
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date']).dt.date
    out = df2.merge(w, on='Date', how='left')
    # engineered flags
    out['is_rain'] = out['precip_mm'].fillna(0) > 0.5
    out['temp_avg'] = out[['temp_max','temp_min']].astype(float).mean(axis=1)
    out['is_cold'] = out['temp_avg'] < 10
    out['is_hot'] = out['temp_avg'] > 30
    return out

def add_holiday_flags(df, country_code='US', date_col='Date'):
    if df is None or len(df)==0:
        return df
    yrs = pd.to_datetime(df[date_col]).dt.year.unique().tolist()
    try:
        holiday_cal = hols.CountryHoliday(country_code, years=yrs)
    except Exception:
        # fallback using common countries mapping
        holiday_cal = hols.CountryHoliday('US', years=yrs)
    df2 = df.copy()
    df2['Holiday_Flag'] = pd.to_datetime(df2[date_col]).dt.date.apply(lambda d: 1 if d in holiday_cal else 0)
    return df2


def batch_enrich_weather(df, sku_location_map_path=None, cache_dir='data/weather_cache'):
    """Batch enrich sales DataFrame with weather for each region/sku mapping.
    sku_location_map_path: CSV with columns SKU_ID, lat, lon (optional).
    If not provided, attempts to use 'Region' column on df and map region->lat/lon via config.
    Returns df with weather columns merged.
    Caches per-location parquet files under cache_dir/{lat}_{lon}.parquet
    """
    import pandas as pd, os
    if sku_location_map_path and os.path.exists(sku_location_map_path):
        mapping = pd.read_csv(sku_location_map_path)
    else:
        # try to infer if df has Location/Region columns
        mapping = None
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date']).dt.date
    # Determine unique SKUs and their lat/lon
    if mapping is not None and 'SKU_ID' in mapping.columns and 'lat' in mapping.columns and 'lon' in mapping.columns:
        map_df = mapping[['SKU_ID','lat','lon']].drop_duplicates()
        merged = df2.merge(map_df, on='SKU_ID', how='left')
    else:
        # if no mapping, assume single location (require lat/lon in df)
        if 'lat' in df2.columns and 'lon' in df2.columns:
            merged = df2.copy()
        else:
            # nothing to do
            return df2
    out_frames = []
    os.makedirs(cache_dir, exist_ok=True)
    grouped = merged.groupby(['lat','lon'], dropna=True)
    for (lat, lon), group in grouped:
        if pd.isna(lat) or pd.isna(lon):
            out_frames.append(group)
            continue
        start = group['Date'].min().strftime('%Y-%m-%d')
        end = group['Date'].max().strftime('%Y-%m-%d')
        cache_file = os.path.join(cache_dir, f"weather_{str(lat).replace('.','_')}_{str(lon).replace('.','_')}.parquet")
        try:
            w = fetch_weather_for_location(float(lat), float(lon), start, end, cache_path=cache_file)
            w['Date'] = pd.to_datetime(w['Date']).dt.date
            g2 = group.merge(w, on='Date', how='left')
            # engineered flags
            g2['is_rain'] = g2['precip_mm'].fillna(0) > 0.5
            g2['temp_avg'] = g2[['temp_max','temp_min']].astype(float).mean(axis=1)
            g2['is_cold'] = g2['temp_avg'] < 10
            g2['is_hot'] = g2['temp_avg'] > 30
            out_frames.append(g2)
        except Exception:
            out_frames.append(group)
    result = pd.concat(out_frames, ignore_index=True)
    return result
