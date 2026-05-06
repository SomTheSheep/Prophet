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
DB_URI = os.environ.get('PROPHETIG_DB_URI', 'postgresql://prophetig_user:prophetig_pass@prophetig-postgres-service:5432/prophetig_db')

METRICS_FILE = '/etc/prophetig/metrics.json'
SIMULATION_FILE = '/etc/prophetig/simulation.json'

def load_configs():
    metrics = {}
    simulation = {}
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
    if os.path.exists(SIMULATION_FILE):
        with open(SIMULATION_FILE, 'r') as f:
            simulation = json.load(f)
    return metrics, simulation

METRICS_CONFIG, SIM_CONFIG = load_configs()

# Same timestamps as your original backup simulation
TIME_START = SIM_CONFIG.get("LIVE_START", "2026-04-13T17:55:09Z")
TIME_END = SIM_CONFIG.get("LIVE_END", "2026-04-20T13:27:15Z")
TRAIN_WINDOW_DAYS = SIM_CONFIG.get("TRAIN_WINDOW_DAYS", 6.72)
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

def add_spike_regressor(df):
    # Mark the 04:00–04:05 window as a known event
    df['cpu_spike'] = 0
    spike_mask = (
        (df['ds'].dt.hour == 4) &
        (df['ds'].dt.minute < 10)  # give a small buffer
    )
    df.loc[spike_mask, 'cpu_spike'] = 1
    return df

def apply_prophet_univariate(df_target, forecast_points):
    df_merged = df_target.rename(columns={'y': 'y'})
    df_merged = df_merged.sort_values('ds').ffill().fillna(0)
    
    # Add the structural binary regressor
    df_merged = add_spike_regressor(df_merged)

    # Load Hyperparameters from Environment Variables
    interval_width = float(os.environ.get('PROPHET_INTERVAL_WIDTH', 0.95))
    changepoint_prior_scale = float(os.environ.get('PROPHET_CHANGEPOINT_PRIOR_SCALE', 0.05))
    seasonality_prior_scale = float(os.environ.get('PROPHET_SEASONALITY_PRIOR_SCALE', 10.0))

    # Use weekly_seasonality if >= ~7 days train window
    use_weekly = TRAIN_WINDOW_DAYS >= 6.5
    m = Prophet(
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=False,
        weekly_seasonality=use_weekly,
        daily_seasonality=True
    )
    
    # Tell Prophet to explicitly account for the boolean regressor column
    m.add_regressor('cpu_spike')

    if len(df_merged) < 2: return pd.DataFrame(), m

    m.fit(df_merged)

    future = m.make_future_dataframe(periods=forecast_points, freq='5min')
    
    # Must apply the exact same regressor logic to the future dataframe before predicting
    future = add_spike_regressor(future)
    
    forecast = m.predict(future)
    return forecast, m

def simulate_historical_data():
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    virtual_now = datetime.strptime(TIME_START, fmt)
    end_dt = datetime.strptime(TIME_END, fmt)
    
    logger.info(f"Starting Prophetig (Univariate) backfill simulator from {virtual_now} to {end_dt}")
    
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
        
        for metric_name, queries in METRICS_CONFIG.items():
            if not queries.get('TARGET_QUERY'):
                continue
                
            logger.info(f"[{metric_name}] Training {TRAIN_WINDOW_DAYS}d history, predicting next {RETRAIN_EVERY_HOURS}h")
            
            df_target_full = query_prometheus(queries['TARGET_QUERY'], train_start, predict_end, '5m')
            
            if df_target_full.empty:
                logger.warning(f"[{metric_name}] No target data found.")
                continue
                
            # Hide the future targets from the training set
            mask = df_target_full['ds'] <= virtual_now
            df_target_train = df_target_full[mask]
            
            if df_target_train.empty:
                continue
                
            # Train the UNIVARIATE model
            forecast, m = apply_prophet_univariate(df_target_train, forecast_points=int(forecast_pts))
            
            if forecast.empty:
                continue
            
            # Extract ONLY the next 6 hour prediction block
            forecast_future = forecast[(forecast['ds'] > virtual_now) & (forecast['ds'] <= next_virtual_now)].copy()
            
            # Merge with ACTUALS to evaluate prediction success
            actuals_future = df_target_full[(df_target_full['ds'] > virtual_now) & (df_target_full['ds'] <= next_virtual_now)].rename(columns={'y': 'actual_value'})
            final_future = pd.merge(forecast_future, actuals_future, on='ds', how='left')
            
            # Prepare DataFrame for SQL
            sql_df = pd.DataFrame({
                'timestamp': final_future['ds'],
                'metric_name': metric_name,
                'actual_value': final_future['actual_value'],
                'yhat': final_future['yhat'],
                'upper': final_future['yhat_upper'],
                'lower': final_future['yhat_lower']
            })
            
            results_df_list.append(sql_df)

        if results_df_list:
            master_df = pd.concat(results_df_list)
            logger.info(f"Writing {len(master_df)} rows to PostgreSQL")
            try:
                master_df.to_sql('prophetig_forecasts', engine, if_exists='append', index=False)
            except Exception as e:
                logger.error(f"SQL Write Failed: {e}")
                
        virtual_now = next_virtual_now

    logger.info("✅ Historical Backfill Complete!")

if __name__ == "__main__":
    time.sleep(5) # Wait for Postgres to spin up
    simulate_historical_data()