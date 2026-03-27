import os, yaml, logging, requests, threading, time
from datetime import datetime, timedelta
from flask import Flask, jsonify, Response, request
import pandas as pd
from prophet import Prophet
from prometheus_client import Gauge, generate_latest, REGISTRY, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

CONFIG_PATH = os.environ.get('CONFIG_PATH', '/app/config.yaml')
PROMETHEUS_TOKEN = os.environ.get('PROMETHEUS_TOKEN', '')

forecasts = {}
anomalies = {}
last_trained = None

anomaly_gauges = {}
forecast_gauges = {}
forecast_lower_gauges = {}
forecast_upper_gauges = {}
anomaly_status_gauges = {}  
actual_value_gauges = {}
future_forecast_gauges = {}

def load_config():
    global config, PROMETHEUS_URL, METRICS_CONFIG, DATA_CONFIG
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    PROMETHEUS_URL = config['prometheus']['url']
    METRICS_CONFIG = config['metrics']
    DATA_CONFIG = config.get('data', {})

load_config()

def get_prometheus_headers():
    headers = {'Accept': 'application/json'}
    if PROMETHEUS_TOKEN: headers['Authorization'] = PROMETHEUS_TOKEN
    return headers

def query_prometheus(query, start_time, end_time, step=300):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {'query': query, 'start': start_time, 'end': end_time, 'step': step}
    try:
        response = requests.get(url, params=params, headers=get_prometheus_headers(), timeout=60)
        if response.status_code == 200 and response.json()['data']['result']:
            df = pd.DataFrame(response.json()['data']['result'][0]['values'], columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'], unit='s')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            return df.dropna()
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
    return pd.DataFrame()

def train_prophet(df, periods=60, seasonality='daily'):
    model = Prophet(daily_seasonality=(seasonality == 'daily'), weekly_seasonality=(seasonality == 'weekly'), yearly_seasonality=False, interval_width=0.95)
    model.fit(df)
    return model.predict(model.make_future_dataframe(periods=periods, freq='5min'))

def detect_anomalies(df, forecast):
    merged = df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
    merged['anomaly'] = (merged['y'] < merged['yhat_lower']) | (merged['y'] > merged['yhat_upper'])
    merged['deviation'] = merged['y'] - merged['yhat']
    merged['deviation_pct'] = (merged['deviation'] / merged['yhat'].abs()) * 100
    return merged

def initialize_metrics():
    for m in METRICS_CONFIG:
        name = m['name']
        if name not in anomaly_gauges:
            anomaly_gauges[name] = Gauge(f'prophet_anomaly_count_{name}', f'Anomaly count')
            forecast_gauges[name] = Gauge(f'prophet_forecast_{name}', f'Latest forecast')
            forecast_lower_gauges[name] = Gauge(f'prophet_lower_{name}', f'Lower bound')
            forecast_upper_gauges[name] = Gauge(f'prophet_upper_{name}', f'Upper bound')
            anomaly_status_gauges[name] = Gauge(f'prophet_anomaly_status_{name}', f'Anomaly status')
            actual_value_gauges[name] = Gauge(f'prophet_actual_{name}', f'Actual value')
            future_forecast_gauges[name] = Gauge(f'prophet_future_forecast_{name}', f'Forecast at end of period')

def train_all_models():
    global last_trained
    lookback = DATA_CONFIG.get('lookback_days', 14)
    end = datetime.utcnow()
    start = end - timedelta(days=lookback)
    
    for m in METRICS_CONFIG:
        name, query, periods, season = m['name'], m['query'], m.get('forecast_periods', 60), m.get('seasonality', 'daily')
        df = query_prometheus(query, start.strftime('%Y-%m-%dT%H:%M:%SZ'), end.strftime('%Y-%m-%dT%H:%M:%SZ'))
        if not df.empty:
            forecast = train_prophet(df, periods, season)
            anomaly_df = detect_anomalies(df, forecast)
            forecasts[name], anomalies[name] = forecast, anomaly_df
            if name in anomaly_gauges: anomaly_gauges[name].set(anomaly_df['anomaly'].sum())
            
            # Note: We remove the static .iloc[-1] set here because metrics() handles dynamic exporting now
            # if name in forecast_gauges: forecast_gauges[name].set(forecast['yhat'].iloc[-1])
            
    last_trained = datetime.utcnow().isoformat()

@app.route('/health')
def health(): return jsonify({'status': 'healthy', 'models_loaded': len(forecasts), 'last_trained': last_trained})

@app.route('/retrain', methods=['POST'])
def retrain():
    load_config()
    initialize_metrics()
    train_all_models()
    return jsonify({'status': 'retrained', 'models': list(forecasts.keys())})

@app.route('/config')
def get_config(): return jsonify({'metrics': [m['name'] for m in METRICS_CONFIG]})

@app.route('/metrics')
def metrics():
    now = datetime.utcnow()
    for name, forecast in forecasts.items():
        if not forecast.empty:
            # Current time evaluation for Anomalies
            idx = abs(forecast['ds'] - now).idxmin()
            forecast_gauges[name].set(forecast.loc[idx, 'yhat'])
            forecast_lower_gauges[name].set(forecast.loc[idx, 'yhat_lower'])
            forecast_upper_gauges[name].set(forecast.loc[idx, 'yhat_upper'])
            
            # Continuous Walk: Dynamically find the matching target in the future array
            # We look for what was predicted roughly 5 days from `now`
            target_future_time = now + timedelta(days=5)
            # Find the closest pre-calculated future tick
            future_idx = abs(forecast['ds'] - target_future_time).idxmin()
            future_val = forecast.loc[future_idx, 'yhat']
            future_forecast_gauges[name].set(future_val)
            
            if name in anomalies and not anomalies[name].empty:
                df = anomalies[name]
                act_idx = abs(df['ds'] - now).idxmin()
                val = df.loc[act_idx, 'y']
                actual_value_gauges[name].set(val)
                anomaly_status_gauges[name].set(1 if (val < forecast.loc[idx, 'yhat_lower'] or val > forecast.loc[idx, 'yhat_upper']) else 0)
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)

def background_retrain_loop(interval_hours=6):
    while True:
        # Sleep exactly N hours before attempting a background retrain
        time.sleep(interval_hours * 3600)
        try:
            logger.info("Executing periodic background retraining of Prophet models...")
            train_all_models()
            logger.info("Periodic background retraining complete.")
        except Exception as e:
            logger.error(f"Periodic background retraining failed: {e}")

if __name__ == '__main__':
    initialize_metrics()
    logger.info("Running initial Prophet model training phase...")
    train_all_models()
    
    # Spin up background loop for rolling 6-hour updates
    threading.Thread(target=background_retrain_loop, args=(6,), daemon=True).start()
    
    app.run(host='0.0.0.0', port=8000)
