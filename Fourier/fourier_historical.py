import os, logging, requests
import numpy as np
import pandas as pd
from datetime import timedelta
from prophet import Prophet
from flask import Flask, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Fetch everything natively from the environment variables defined in deployment.yaml
PROM_URL = os.environ.get('FLT_PROM_URL', 'https://prometheus.sitopflab03.otv-staging.com')
PROM_TOKEN = os.environ.get('FLT_PROM_ACCESS_TOKEN', '')
TARGET_QUERY = os.environ.get('FLT_TARGET_QUERY', '')
REGRESSOR_CPU = os.environ.get('FLT_REGRESSOR_CPU', '')
REGRESSOR_RPS = os.environ.get('FLT_REGRESSOR_RPS', '')
REGRESSOR_5XX = os.environ.get('FLT_REGRESSOR_5XX', '')

START_TIME_STR = os.environ.get('FLT_DATA_START_TIME', '2026-02-16T12:00:00Z')
END_TIME_STR = os.environ.get('FLT_DATA_END_TIME', '2026-03-04T05:21:16Z')

def get_prometheus_headers():
    headers = {'Accept': 'application/json'}
    if PROM_TOKEN:
        headers['Authorization'] = f"Bearer {PROM_TOKEN}" if not PROM_TOKEN.startswith('Bearer ') else PROM_TOKEN
    return headers

def query_prometheus(query, start_time, end_time, step='10m'):
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

def apply_prophet_multivariate(df_target, df_cpu, df_rps, df_5xx, forecast_points=36):
    df_merged = df_target.rename(columns={'y': 'y'})
    
    if not df_cpu.empty:
        df_cpu = df_cpu.rename(columns={'y': 'cpu'})
        df_merged = pd.merge(df_merged, df_cpu, on='ds', how='outer')
    if not df_rps.empty:
        df_rps = df_rps.rename(columns={'y': 'rps'})
        df_merged = pd.merge(df_merged, df_rps, on='ds', how='outer')
    if not df_5xx.empty:
        df_5xx = df_5xx.rename(columns={'y': 'err5xx'})
        df_merged = pd.merge(df_merged, df_5xx, on='ds', how='outer')

    df_merged = df_merged.sort_values('ds').ffill().fillna(0)

    m = Prophet(interval_width=0.95, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
    
    if 'cpu' in df_merged.columns: m.add_regressor('cpu')
    if 'rps' in df_merged.columns: m.add_regressor('rps')
    if 'err5xx' in df_merged.columns: m.add_regressor('err5xx')

    m.fit(df_merged)

    future = m.make_future_dataframe(periods=forecast_points, freq='10min')
    
    # Forward-fill regressors into the future using last known values
    if 'cpu' in df_merged.columns:
        future['cpu'] = df_merged['cpu'].iloc[-1]
    if 'rps' in df_merged.columns:
        future['rps'] = df_merged['rps'].iloc[-1]
    if 'err5xx' in df_merged.columns:
        future['err5xx'] = df_merged['err5xx'].iloc[-1]

    forecast = m.predict(future)
    return df_merged, forecast, m

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multivariate Prophet Forecast</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
</head>
<body style="font-family: Arial; padding: 20px;">
    <h1>Multivariate Prophet Anomaly Forecast</h1>
    <div style="background:#f5f5f5;padding:15px;border-radius:8px;margin-bottom:20px;word-break:break-all;">
        <p><strong>Target:</strong> P90 Latency</p>
        <p><strong>Regressors:</strong> CPU, RPS, 5xx Rate</p>
        <p><strong>Forecast:</strong> 6 Hours forward</p>
    </div>
    
    <div style="width: 90%; margin: auto;">
        <canvas id="prophetChart"></canvas>
    </div>
    <script>
        fetch('/api/data')
            .then(res => res.json())
            .then(data => {
                if(data.error) {
                    alert(data.error);
                    return;
                }
                const ctx = document.getElementById('prophetChart').getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.labels,
                        datasets: [
                            { label: 'Latency (y)', data: data.actual, borderColor: 'black', fill: false, pointRadius: 0, borderWidth: 1, yAxisID: 'y' },
                            { label: 'Forecast Baseline (yhat)', data: data.yhat, borderColor: 'blue', borderDash: [5, 5], fill: false, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
                            { label: 'Upper Bound', data: data.yhat_upper, borderColor: 'rgba(255, 99, 132, 0.5)', fill: false, pointRadius: 0, borderWidth: 1, yAxisID: 'y' },
                            { label: 'Lower Bound', data: data.yhat_lower, borderColor: 'rgba(255, 99, 132, 0.5)', fill: '-1', backgroundColor: 'rgba(255, 99, 132, 0.2)', pointRadius: 0, borderWidth: 1, yAxisID: 'y' },
                            { label: 'RPS (Regressor)', data: data.rps, borderColor: 'orange', fill: false, pointRadius: 0, borderWidth: 1, yAxisID: 'y1' }
                        ]
                    },
                    options: {
                        responsive: true,
                        interaction: { mode: 'index', intersect: false },
                        scales: { 
                            x: { type: 'time', time: { unit: 'hour' } },
                            y: { type: 'linear', display: true, position: 'left', title: {display: true, text: 'Latency / Yhat'} },
                            y1: { type: 'linear', display: true, position: 'right', grid: {drawOnChartArea: false}, title: {display: true, text: 'RPS Traffic'} }
                        }
                    }
                });
            });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    df_target = query_prometheus(TARGET_QUERY, START_TIME_STR, END_TIME_STR, step='10m')
    if df_target.empty:
        return jsonify({'error': 'No target data retrieved'})

    df_cpu = query_prometheus(REGRESSOR_CPU, START_TIME_STR, END_TIME_STR, step='10m')
    df_rps = query_prometheus(REGRESSOR_RPS, START_TIME_STR, END_TIME_STR, step='10m')
    df_5xx = query_prometheus(REGRESSOR_5XX, START_TIME_STR, END_TIME_STR, step='10m')
        
    forecast_points = 36 # 6 hours at 10-min intervals
    df_merged, forecast, m = apply_prophet_multivariate(df_target, df_cpu, df_rps, df_5xx, forecast_points=forecast_points)
    
    # Pad actuals with None for forecast points so arrays align
    actual_vals = df_merged['y'].tolist() + [None] * forecast_points
    rps_vals = df_merged['rps'].tolist() + [df_merged['rps'].iloc[-1]] * forecast_points if 'rps' in df_merged.columns else []

    return jsonify({
        'labels': [t.strftime('%Y-%m-%dT%H:%M:%S') for t in forecast['ds']],
        'actual': actual_vals,
        'yhat': forecast['yhat'].tolist(),
        'yhat_upper': forecast['yhat_upper'].tolist(),
        'yhat_lower': forecast['yhat_lower'].tolist(),
        'rps': rps_vals
    })

if __name__ == '__main__':
    # Listen on port 8080 exactly matching what K8s deployment expects
    app.run(host='0.0.0.0', port=8080, use_reloader=False)
