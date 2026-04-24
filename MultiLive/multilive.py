import os
import logging
import requests
import json
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from flask import Flask, jsonify, request
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Add prometheus wsgi middleware to route /metrics
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

PROM_URL = os.environ.get('FLT_PROM_URL', 'https://prometheus.sitopflab03.otv-staging.com')
PROM_TOKEN = os.environ.get('FLT_PROM_ACCESS_TOKEN', '')

CONFIG_FILE = '/etc/multilive/endpoints.json'
ENDPOINTS_CONFIG = {}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        ENDPOINTS_CONFIG = json.load(f)
else:
    # Dummy dev endpoints config
    ENDPOINTS_CONFIG = {"widevine": {"TARGET_QUERY": "","REGRESSOR_CPU": "","REGRESSOR_RPS": "","REGRESSOR_5XX": ""}}

# --- PROMETHEUS GAUGE METRICS ---
LATENCY_BASELINE = Gauge('multilive_predicted_latency_baseline', 'Predicted Prophet Latency Baseline (ms)', ['endpoint', 'type', 'horizon'])
REGRESSOR_IMPACT = Gauge('multilive_regressor_impact_ms', 'Impact of Regressors (ms)', ['endpoint', 'regressor', 'horizon'])
TRAINING_TIME = Gauge('multilive_model_training_duration_seconds', 'Execution time for Prophet Training', ['endpoint'])
GLOBAL_STATUS = Gauge('multilive_training_last_success', 'Unix timestamp of last successful background model training')
# --------------------------------

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

def apply_prophet_multivariate(df_target, df_cpu, df_rps, df_5xx, forecast_points=72, freq='5min'):
    df_merged = df_target.rename(columns={'y': 'y'})
    
    if not df_cpu.empty: df_merged = pd.merge(df_merged, df_cpu.rename(columns={'y': 'cpu'}), on='ds', how='outer')
    if not df_rps.empty: df_merged = pd.merge(df_merged, df_rps.rename(columns={'y': 'rps'}), on='ds', how='outer')
    if not df_5xx.empty: df_merged = pd.merge(df_merged, df_5xx.rename(columns={'y': 'err5xx'}), on='ds', how='outer')

    df_merged = df_merged.sort_values('ds').ffill().fillna(0)

    m = Prophet(interval_width=0.95, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True)
    
    if 'cpu' in df_merged.columns: m.add_regressor('cpu')
    if 'rps' in df_merged.columns: m.add_regressor('rps')
    if 'err5xx' in df_merged.columns: m.add_regressor('err5xx')

    if len(df_merged) < 2: return df_merged, pd.DataFrame(), m

    m.fit(df_merged)

    future = m.make_future_dataframe(periods=forecast_points, freq=freq)
    
    cols_to_merge = ['ds']
    if 'cpu' in df_merged.columns: cols_to_merge.append('cpu')
    if 'rps' in df_merged.columns: cols_to_merge.append('rps')
    if 'err5xx' in df_merged.columns: cols_to_merge.append('err5xx')
        
    if len(cols_to_merge) > 1:
        future = pd.merge(future, df_merged[cols_to_merge], on='ds', how='left')
        
    future = future.ffill()
    forecast = m.predict(future)
    return df_merged, forecast, m

def extract_metric_val(df_row, column_name):
    return float(df_row[column_name]) if column_name in df_row else 0.0

def train_all_endpoints():
    logger.info("Starting background Prophet training for all endpoints...")
    end_time = datetime.utcnow().timestamp()
    start_time = (datetime.utcnow() - timedelta(days=14)).timestamp()

    for ep, queries in ENDPOINTS_CONFIG.items():
        if not queries.get('TARGET_QUERY'):
            continue
            
        t0 = time.time()
        logger.info(f"Training [{ep}] using 14-day history")
        
        df_target = query_prometheus(queries['TARGET_QUERY'], start_time, end_time, '5m')
        if df_target.empty:
            logger.warning(f"No target data for [{ep}], skipping...")
            continue
            
        df_cpu = query_prometheus(queries.get('REGRESSOR_CPU', ''), start_time, end_time, '5m')
        df_rps = query_prometheus(queries.get('REGRESSOR_RPS', ''), start_time, end_time, '5m')
        df_5xx = query_prometheus(queries.get('REGRESSOR_5XX', ''), start_time, end_time, '5m')
        
        # 72 periods * 5 mins = 6 hours
        df_merged, forecast, m = apply_prophet_multivariate(df_target, df_cpu, df_rps, df_5xx, forecast_points=72, freq='5min')
        
        if forecast.empty:
            continue
            
        # len(df_merged) represents the boundary between historical data and future predictions
        live_idx = len(df_merged) - 1
        now_row = forecast.iloc[live_idx] if live_idx >= 0 else forecast.iloc[-73]
        future_row = forecast.iloc[-1]
        
        # Metrics for horizon='now'
        LATENCY_BASELINE.labels(endpoint=ep, type='yhat', horizon='now').set(extract_metric_val(now_row, 'yhat'))
        LATENCY_BASELINE.labels(endpoint=ep, type='lower', horizon='now').set(extract_metric_val(now_row, 'yhat_lower'))
        LATENCY_BASELINE.labels(endpoint=ep, type='upper', horizon='now').set(extract_metric_val(now_row, 'yhat_upper'))
        
        # Metrics for horizon='plus_6h'
        LATENCY_BASELINE.labels(endpoint=ep, type='yhat', horizon='plus_6h').set(extract_metric_val(future_row, 'yhat'))
        LATENCY_BASELINE.labels(endpoint=ep, type='lower', horizon='plus_6h').set(extract_metric_val(future_row, 'yhat_lower'))
        LATENCY_BASELINE.labels(endpoint=ep, type='upper', horizon='plus_6h').set(extract_metric_val(future_row, 'yhat_upper'))
        
        if 'cpu' in forecast.columns:
            REGRESSOR_IMPACT.labels(endpoint=ep, regressor='cpu', horizon='now').set(extract_metric_val(now_row, 'cpu'))
            REGRESSOR_IMPACT.labels(endpoint=ep, regressor='cpu', horizon='plus_6h').set(extract_metric_val(future_row, 'cpu'))
        if 'rps' in forecast.columns:
            REGRESSOR_IMPACT.labels(endpoint=ep, regressor='rps', horizon='now').set(extract_metric_val(now_row, 'rps'))
            REGRESSOR_IMPACT.labels(endpoint=ep, regressor='rps', horizon='plus_6h').set(extract_metric_val(future_row, 'rps'))
        if 'err5xx' in forecast.columns:
            REGRESSOR_IMPACT.labels(endpoint=ep, regressor='5xx', horizon='now').set(extract_metric_val(now_row, 'err5xx'))
            REGRESSOR_IMPACT.labels(endpoint=ep, regressor='5xx', horizon='plus_6h').set(extract_metric_val(future_row, 'err5xx'))
            
        exec_time = time.time() - t0
        TRAINING_TIME.labels(endpoint=ep).set(exec_time)
        logger.info(f"Finished [{ep}] in {exec_time:.2f}s")
        
    GLOBAL_STATUS.set(time.time())
    logger.info("Background run complete.")

def background_loop():
    time.sleep(5)
    while True:
        try:
            train_all_endpoints()
        except Exception as e:
            logger.error(f"Error in background loop: {e}")
        logger.info("Sleeping background daemon for 30 minutes...")
        time.sleep(1800)

@app.route('/retrain', methods=['POST'])
def handle_retrain():
    t = threading.Thread(target=train_all_endpoints, daemon=True)
    t.start()
    return jsonify({"status": "training initiated in background", "timestamp": time.time()})

if __name__ == '__main__':
    logger.info("Spinning up background Prophet training daemon...")
    bg_thread = threading.Thread(target=background_loop, daemon=True)
    bg_thread.start()
    app.run(host='0.0.0.0', port=8080)
