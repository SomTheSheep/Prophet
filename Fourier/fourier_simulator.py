import os
import logging
import requests
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROM_URL = os.environ.get('FLT_PROM_URL', 'https://prometheus.sitopflab03.otv-staging.com')
PROM_TOKEN = os.environ.get('FLT_PROM_ACCESS_TOKEN', '')
DB_URI = os.environ.get('FOURIER_DB_URI', 'postgresql://fourier_user:fourier_pass@fourier-postgres-service:5432/fourier_db')

ENDPOINTS_FILE = '/etc/fourier/endpoints.json'
SIMULATION_FILE = '/etc/fourier/simulation.json'

def load_configs():
    endpoints = {}
    simulation = {}
    if os.path.exists(ENDPOINTS_FILE):
        with open(ENDPOINTS_FILE, 'r') as f:
            endpoints = json.load(f)
    if os.path.exists(SIMULATION_FILE):
        with open(SIMULATION_FILE, 'r') as f:
            simulation = json.load(f)
    return endpoints, simulation

ENDPOINTS_CONFIG, SIM_CONFIG = load_configs()

# Default to roughly recent timestamps if missing
TIME_START = SIM_CONFIG.get("LIVE_START", "2024-04-10T12:00:00Z")
TIME_END = SIM_CONFIG.get("LIVE_END", "2024-04-17T12:00:00Z")
TRAIN_WINDOW_DAYS = SIM_CONFIG.get("TRAIN_WINDOW_DAYS", 7)
RETRAIN_EVERY_HOURS = SIM_CONFIG.get("RETRAIN_EVERY_HOURS", 6)

engine = create_engine(DB_URI)

def get_prometheus_headers():
    headers = {'Accept': 'application/json'}
    if PROM_TOKEN:
        headers['Authorization'] = f"Bearer {PROM_TOKEN}" if not PROM_TOKEN.startswith('Bearer ') else PROM_TOKEN
    return headers

def query_prometheus(query, start_time, end_time, step='5m'):
    if not query:
        return pd.DataFrame()
    query_url = f"{PROM_URL}/api/v1/query_range"
    params = {'query': query, 'start': start_time, 'end': end_time, 'step': step}
    try:
        response = requests.get(query_url, params=params, headers=get_prometheus_headers(), verify=False, timeout=120)
        data = response.json()
        if data.get('status') == 'success' and len(data.get('data', {}).get('result', [])) > 0:
            df = pd.DataFrame(data['data']['result'][0]['values'], columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'], unit='s')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            return df
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
    return pd.DataFrame()

def apply_prophet_multivariate(df_target, df_cpu, df_rps, df_5xx, forecast_points):
    df_merged = df_target.rename(columns={'y': 'y'})
    
    if not df_cpu.empty: df_merged = pd.merge(df_merged, df_cpu.rename(columns={'y': 'cpu'}), on='ds', how='outer')
    if not df_rps.empty: df_merged = pd.merge(df_merged, df_rps.rename(columns={'y': 'rps'}), on='ds', how='outer')
    if not df_5xx.empty: df_merged = pd.merge(df_merged, df_5xx.rename(columns={'y': 'err5xx'}), on='ds', how='outer')

    df_merged = df_merged.sort_values('ds').ffill().fillna(0)

    # Use weekly_seasonality if >= 7 days train window
    use_weekly = TRAIN_WINDOW_DAYS >= 7
    m = Prophet(interval_width=0.95, yearly_seasonality=False, weekly_seasonality=use_weekly, daily_seasonality=True)
    
    if 'cpu' in df_merged.columns: m.add_regressor('cpu')
    if 'rps' in df_merged.columns: m.add_regressor('rps')
    if 'err5xx' in df_merged.columns: m.add_regressor('err5xx')

    if len(df_merged) < 2: return pd.DataFrame(), m

    m.fit(df_merged)

    future = m.make_future_dataframe(periods=forecast_points, freq='5min')
    
    # Merge regressors back in so prediction has actual latest regressor data
    cols_to_merge = ['ds']
    if 'cpu' in df_merged.columns: cols_to_merge.append('cpu')
    if 'rps' in df_merged.columns: cols_to_merge.append('rps')
    if 'err5xx' in df_merged.columns: cols_to_merge.append('err5xx')
        
    if len(cols_to_merge) > 1:
        future = pd.merge(future, df_merged[cols_to_merge], on='ds', how='left')
        
    future = future.ffill()
    forecast = m.predict(future)
    return forecast, m

def simulate_historical_data():
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    virtual_now = datetime.strptime(TIME_START, fmt)
    end_dt = datetime.strptime(TIME_END, fmt)
    
    logger.info(f"Starting historical backfill simulator from {virtual_now} to {end_dt}")
    
    while virtual_now < end_dt:
        logger.info(f"--- Processing Virtual Time Window: {virtual_now} ---")
        
        train_start = (virtual_now - timedelta(days=TRAIN_WINDOW_DAYS)).timestamp()
        train_end = virtual_now.timestamp()
        
        # Add 6 hours to current "live" cursor
        next_virtual_now = virtual_now + timedelta(hours=RETRAIN_EVERY_HOURS)
        if next_virtual_now > end_dt:
            next_virtual_now = end_dt
            
        predict_end = next_virtual_now.timestamp()
        forecast_pts = int((predict_end - train_end) / 300) + 1  # 5 min steps = 300s
        
        results_df_list = []
        
        for ep, queries in ENDPOINTS_CONFIG.items():
            if not queries.get('TARGET_QUERY'):
                continue
                
            logger.info(f"[{ep}] Training {TRAIN_WINDOW_DAYS}d history, predicting next {RETRAIN_EVERY_HOURS}h")
            
            # Note: For the actual live implementation we must fetch metric up to `predict_end`
            # So df_target handles train + exact future regressors.
            df_target_full = query_prometheus(queries['TARGET_QUERY'], train_start, predict_end, '5m')
            
            if df_target_full.empty:
                logger.warning(f"[{ep}] No target data found.")
                continue
                
            df_cpu_full = query_prometheus(queries.get('REGRESSOR_CPU', ''), train_start, predict_end, '5m')
            df_rps_full = query_prometheus(queries.get('REGRESSOR_RPS', ''), train_start, predict_end, '5m')
            df_5xx_full = query_prometheus(queries.get('REGRESSOR_5XX', ''), train_start, predict_end, '5m')
            
            # Split into training data strictly up to `virtual_now`
            mask = df_target_full['ds'] <= virtual_now
            df_target_train = df_target_full[mask]
            df_cpu_train = df_cpu_full[df_cpu_full['ds'] <= virtual_now] if not df_cpu_full.empty else df_cpu_full
            df_rps_train = df_rps_full[df_rps_full['ds'] <= virtual_now] if not df_rps_full.empty else df_rps_full
            df_5xx_train = df_5xx_full[df_5xx_full['ds'] <= virtual_now] if not df_5xx_full.empty else df_5xx_full
            
            if df_target_train.empty:
                continue
                
            # Train the model over history, outputting forecast that uses future regressor values (by supplying full data)
            forecast, m = apply_prophet_multivariate(df_target_train, df_cpu_full, df_rps_full, df_5xx_full, forecast_points=0)
            
            if forecast.empty:
                continue
            
            # Now we extract ONLY the `future` span for our SQL Database
            forecast_future = forecast[(forecast['ds'] > virtual_now) & (forecast['ds'] <= next_virtual_now)].copy()
            
            # Merge with ACTUALS to store in DB
            actuals_future = df_target_full[(df_target_full['ds'] > virtual_now) & (df_target_full['ds'] <= next_virtual_now)].rename(columns={'y': 'actual_p50'})
            final_future = pd.merge(forecast_future, actuals_future, on='ds', how='left')
            
            # Prepare DataFrame for SQL
            sql_df = pd.DataFrame({
                'timestamp': final_future['ds'],
                'endpoint': ep,
                'actual_p50': final_future['actual_p50'],
                'yhat': final_future['yhat'],
                'upper': final_future['yhat_upper'],
                'lower': final_future['yhat_lower'],
                'cpu_impact': final_future.get('cpu', 0.0),
                'rps_impact': final_future.get('rps', 0.0),
                'err5xx_impact': final_future.get('err5xx', 0.0)
            })
            
            results_df_list.append(sql_df)

        if results_df_list:
            master_df = pd.concat(results_df_list)
            logger.info(f"Writing {len(master_df)} rows to PostgreSQL")
            try:
                master_df.to_sql('simulated_forecasts', engine, if_exists='append', index=False)
            except Exception as e:
                logger.error(f"SQL Write Failed: {e}")
                
        virtual_now = next_virtual_now

    logger.info("✅ Historical Backfill Complete!")

if __name__ == "__main__":
    time.sleep(5) # Wait for Postgres to spin up
    simulate_historical_data()
