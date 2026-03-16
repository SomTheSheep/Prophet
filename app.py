import os
import yaml
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, Response, request
import pandas as pd
import requests
from prophet import Prophet
from prometheus_client import Gauge, generate_latest, REGISTRY, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load config
CONFIG_PATH = os.environ.get('CONFIG_PATH', '/app/config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROMETHEUS_URL = config['prometheus']['url']
METRICS_CONFIG = config['metrics']
DATA_CONFIG = config.get('data', {})

# Get auth token from environment (mounted from secret)
PROMETHEUS_TOKEN = os.environ.get('PROMETHEUS_TOKEN', '')

# Storage for results
forecasts = {}
anomalies = {}
last_trained = None

# Prometheus metrics
anomaly_gauges = {}
forecast_gauges = {}
forecast_lower_gauges = {}
forecast_upper_gauges = {}
anomaly_status_gauges = {}  
actual_value_gauges = {}

def get_prometheus_headers():
    """Build headers for Prometheus requests."""
    headers = {'Accept': 'application/json'}
    if PROMETHEUS_TOKEN:
        headers['Authorization'] = PROMETHEUS_TOKEN
    return headers

def query_prometheus(query, start_time, end_time, step=300):
    """Query Prometheus for historical data."""
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {
        'query': query,
        'start': start_time,
        'end': end_time,
        'step': step
    }
    try:
        response = requests.get(url, params=params, headers=get_prometheus_headers(), timeout=60)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'success' and data['data']['result']:
            values = data['data']['result'][0]['values']
            df = pd.DataFrame(values, columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'], unit='s')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df = df.dropna()
            return df
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
    return pd.DataFrame()

def train_prophet(df, periods=60, seasonality='daily'):
    """Train Prophet model and generate forecast."""
    model = Prophet(
        daily_seasonality=(seasonality == 'daily'),
        weekly_seasonality=(seasonality == 'weekly'),
        yearly_seasonality=False,
        interval_width=0.95
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='5min')
    forecast = model.predict(future)
    return forecast

def detect_anomalies(df, forecast):
    """Detect anomalies where actual values fall outside prediction interval."""
    merged = df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
    merged['anomaly'] = (merged['y'] < merged['yhat_lower']) | (merged['y'] > merged['yhat_upper'])
    merged['deviation'] = merged['y'] - merged['yhat']
    merged['deviation_pct'] = (merged['deviation'] / merged['yhat'].abs()) * 100
    return merged

def initialize_metrics():
    """Initialize Prometheus gauges for each metric."""
    for metric_config in METRICS_CONFIG:
        name = metric_config['name']
        anomaly_gauges[name] = Gauge(f'prophet_anomaly_count_{name}', f'Anomaly count for {name}')
        forecast_gauges[name] = Gauge(f'prophet_forecast_{name}', f'Latest forecast for {name}')
        forecast_lower_gauges[name] = Gauge(f'prophet_lower_{name}', f'Lower bound for {name}')
        forecast_upper_gauges[name] = Gauge(f'prophet_upper_{name}', f'Upper bound for {name}')
        anomaly_status_gauges[name] = Gauge(f'prophet_anomaly_status_{name}', f'Anomaly status for {name}')
        actual_value_gauges[name] = Gauge(f'prophet_actual_{name}', f'Actual value for {name}')
def get_time_range():
    """Calculate time range based on config."""
    global last_trained
    
    use_historical = DATA_CONFIG.get('use_historical', False)
    
    if use_historical:
        start_time = DATA_CONFIG.get('start_time')
        end_time = DATA_CONFIG.get('end_time')
    else:
        lookback_days = DATA_CONFIG.get('lookback_days', 14)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)
        start_time = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    last_trained = datetime.utcnow().isoformat()
    return start_time, end_time

def train_all_models():
    """Train models for all configured metrics."""
    start_time, end_time = get_time_range()
    
    logger.info(f"Training with data from {start_time} to {end_time}")
    logger.info(f"Prometheus URL: {PROMETHEUS_URL}")
    logger.info(f"Auth configured: {'Yes' if PROMETHEUS_TOKEN else 'No'}")
    
    for metric_config in METRICS_CONFIG:
        name = metric_config['name']
        query = metric_config['query']
        periods = metric_config.get('forecast_periods', 60)
        seasonality = metric_config.get('seasonality', 'daily')
        
        logger.info(f"Training model for {name}...")
        
        df = query_prometheus(query, start_time, end_time)
        if df.empty:
            logger.warning(f"No data for {name}")
            continue
        
        logger.info(f"Got {len(df)} data points for {name}")
        
        forecast = train_prophet(df, periods, seasonality)
        anomaly_df = detect_anomalies(df, forecast)
        
        forecasts[name] = forecast
        anomalies[name] = anomaly_df
        
        anomaly_count = anomaly_df['anomaly'].sum()
        if name in anomaly_gauges:
            anomaly_gauges[name].set(anomaly_count)
        if name in forecast_gauges and not forecast.empty:
            forecast_gauges[name].set(forecast['yhat'].iloc[-1])
        
        logger.info(f"Completed {name}: {len(df)} points, {anomaly_count} anomalies")

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(forecasts),
        'last_trained': last_trained,
        'auth_configured': bool(PROMETHEUS_TOKEN)
    })

@app.route('/forecasts')
def get_forecasts():
    summary = {}
    for name, forecast in forecasts.items():
        summary[name] = {
            'periods': len(forecast),
            'last_forecast': forecast['yhat'].iloc[-1] if not forecast.empty else None,
            'forecast_end': forecast['ds'].iloc[-1].isoformat() if not forecast.empty else None
        }
    return jsonify(summary)

@app.route('/anomalies')
def get_all_anomalies():
    summary = {}
    for name, df in anomalies.items():
        anomaly_rows = df[df['anomaly'] == True]
        summary[name] = {
            'total_points': len(df),
            'anomaly_count': len(anomaly_rows),
            'anomaly_percentage': round(len(anomaly_rows) / len(df) * 100, 2) if len(df) > 0 else 0
        }
    return jsonify(summary)

@app.route('/anomalies/<metric_name>')
def get_metric_anomalies(metric_name):
    if metric_name not in anomalies:
        return jsonify({'error': f'Metric {metric_name} not found'}), 404
    
    df = anomalies[metric_name]
    anomaly_rows = df[df['anomaly'] == True]
    
    return jsonify({
        'metric': metric_name,
        'total_points': len(df),
        'anomaly_count': len(anomaly_rows),
        'anomalies': anomaly_rows[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper', 'deviation_pct']].to_dict(orient='records')
    })

@app.route('/chart/<metric_name>')
def chart(metric_name):
    if metric_name not in forecasts or metric_name not in anomalies:
        return f"Metric {metric_name} not found", 404
    
    forecast = forecasts[metric_name]
    anomaly_df = anomalies[metric_name]
    
    dates = forecast['ds'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    yhat = forecast['yhat'].tolist()
    yhat_lower = forecast['yhat_lower'].tolist()
    yhat_upper = forecast['yhat_upper'].tolist()
    
    actual_data = anomaly_df.set_index('ds')['y'].reindex(forecast['ds']).tolist()
    anomaly_points = anomaly_df[anomaly_df['anomaly'] == True]
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prophet Forecast - {metric_name}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h1 {{ color: #00d4ff; }}
            .stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
            .stat-box {{ background: #16213e; padding: 15px; border-radius: 8px; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
            .info {{ background: #16213e; padding: 10px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; }}
            canvas {{ background: #16213e; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Prophet Forecast: {metric_name}</h1>
            <div class="info">
                Last trained: {last_trained} | Data points: {len(anomaly_df)} | 
                <a href="/retrain" style="color: #00d4ff;" onclick="fetch('/retrain', {{method: 'POST'}}).then(() => location.reload()); return false;">Retrain Now</a>
            </div>
            <div class="stats">
                <div class="stat-box">
                    <div>Total Data Points</div>
                    <div class="stat-value">{len(anomaly_df)}</div>
                </div>
                <div class="stat-box">
                    <div>Anomalies Detected</div>
                    <div class="stat-value" style="color: #ff6b6b;">{len(anomaly_points)}</div>
                </div>
                <div class="stat-box">
                    <div>Anomaly Rate</div>
                    <div class="stat-value">{round(len(anomaly_points)/len(anomaly_df)*100, 2) if len(anomaly_df) > 0 else 0}%</div>
                </div>
            </div>
            <canvas id="chart" height="100"></canvas>
        </div>
        <script>
            const ctx = document.getElementById('chart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {dates},
                    datasets: [
                        {{
                            label: 'Actual',
                            data: {actual_data},
                            borderColor: '#00d4ff',
                            backgroundColor: 'transparent',
                            pointRadius: 1,
                            borderWidth: 1
                        }},
                        {{
                            label: 'Forecast',
                            data: {yhat},
                            borderColor: '#4ade80',
                            backgroundColor: 'transparent',
                            pointRadius: 0,
                            borderWidth: 2,
                            borderDash: [5, 5]
                        }},
                        {{
                            label: 'Upper Bound',
                            data: {yhat_upper},
                            borderColor: 'rgba(255, 107, 107, 0.3)',
                            backgroundColor: 'rgba(255, 107, 107, 0.1)',
                            pointRadius: 0,
                            borderWidth: 1,
                            fill: '+1'
                        }},
                        {{
                            label: 'Lower Bound',
                            data: {yhat_lower},
                            borderColor: 'rgba(255, 107, 107, 0.3)',
                            backgroundColor: 'transparent',
                            pointRadius: 0,
                            borderWidth: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    interaction: {{ intersect: false, mode: 'index' }},
                    scales: {{
                        x: {{ ticks: {{ maxTicksLimit: 20, color: '#888' }}, grid: {{ color: '#333' }} }},
                        y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }}
                    }},
                    plugins: {{ legend: {{ labels: {{ color: '#eee' }} }} }}
                }}
            }});
        </script>
    </body>
    </html>
    '''
    return Response(html, mimetype='text/html')

@app.route('/export/anomalies/<metric_name>')
def export_anomalies(metric_name):
    if metric_name not in anomalies:
        return jsonify({'error': f'Metric {metric_name} not found'}), 404
    
    df = anomalies[metric_name]
    csv = df.to_csv(index=False)
    return Response(csv, mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename={metric_name}_anomalies.csv'})

@app.route('/export/forecast/<metric_name>')
def export_forecast(metric_name):
    if metric_name not in forecasts:
        return jsonify({'error': f'Metric {metric_name} not found'}), 404
    
    df = forecasts[metric_name]
    csv = df.to_csv(index=False)
    return Response(csv, mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename={metric_name}_forecast.csv'})

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics with live forecast for current time."""
    now = datetime.utcnow()

    for name, forecast in forecasts.items():
        if not forecast.empty:
            # Find prediction closest to current time
            time_diffs = abs(forecast['ds'] - now)
            closest_idx = time_diffs.idxmin()
            
            forecast_gauges[name].set(forecast.loc[closest_idx, 'yhat'])
            forecast_lower_gauges[name].set(forecast.loc[closest_idx, 'yhat_lower'])
            forecast_upper_gauges[name].set(forecast.loc[closest_idx, 'yhat_upper'])

            # Get actual value and check anomaly status
            if name in anomalies and not anomalies[name].empty:
                df = anomalies[name]
                time_diffs_actual = abs(df['ds'] - now)
                closest_actual_idx = time_diffs_actual.idxmin()
                actual_value = df.loc[closest_actual_idx, 'y']

                actual_value_gauges[name].set(actual_value)  # <-- ADD THIS LINE

                lower = forecast.loc[closest_idx, 'yhat_lower']
                upper = forecast.loc[closest_idx, 'yhat_upper']
                is_anomaly = 1 if (actual_value < lower or actual_value > upper) else 0
                anomaly_status_gauges[name].set(is_anomaly)
                
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)
@app.route('/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining with fresh data."""
    train_all_models()
    return jsonify({
        'status': 'retrained',
        'models': list(forecasts.keys()),
        'last_trained': last_trained
    })

@app.route('/config')
def get_config():
    """Show current configuration (no secrets)."""
    return jsonify({
        'prometheus_url': PROMETHEUS_URL,
        'use_historical': DATA_CONFIG.get('use_historical', False),
        'lookback_days': DATA_CONFIG.get('lookback_days', 14),
        'metrics': [m['name'] for m in METRICS_CONFIG],
        'auth_configured': bool(PROMETHEUS_TOKEN)
    })

if __name__ == '__main__':
    initialize_metrics()
    logger.info("Starting initial model training...")
    train_all_models()
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=8000)
