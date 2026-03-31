import os, logging, requests
import numpy as np
import pandas as pd
from datetime import timedelta
from flask import Flask, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Fetch everything natively from the environment variables defined in deployment.yaml
PROM_URL = os.environ.get('FLT_PROM_URL', 'https://prometheus.sitopflab03.otv-staging.com')
PROM_TOKEN = os.environ.get('FLT_PROM_ACCESS_TOKEN', '')
METRIC_QUERY = os.environ.get('FLT_METRICS_LIST', '')
START_TIME_STR = os.environ.get('FLT_DATA_START_TIME', '2026-02-16T12:00:00Z')
END_TIME_STR = os.environ.get('FLT_DATA_END_TIME', '2026-03-04T05:21:16Z')

def get_prometheus_headers():
    headers = {'Accept': 'application/json'}
    if PROM_TOKEN:
        headers['Authorization'] = f"Bearer {PROM_TOKEN}" if not PROM_TOKEN.startswith('Bearer ') else PROM_TOKEN
    return headers

def query_prometheus(start_time, end_time, step='15m'):
    if not METRIC_QUERY:
        logger.error("No FLT_METRICS_LIST query provided in environment variables")
        return pd.DataFrame()
        
    query_url = f"{PROM_URL}/api/v1/query_range"
    params = {'query': METRIC_QUERY, 'start': start_time, 'end': end_time, 'step': step}
    try:
        response = requests.get(query_url, params=params, headers=get_prometheus_headers(), verify=False, timeout=120)
        data = response.json()
        if data.get('status') == 'success' and len(data.get('data', {}).get('result', [])) > 0:
            df = pd.DataFrame(data['data']['result'][0]['values'], columns=['timestamp', 'y'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df['y'] = df['y'].ffill().fillna(0) # Handle NaN
            return df
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
    return pd.DataFrame()

def apply_fourier_transform(df, num_harmonics=10, forecast_points=480):
    y = df['y'].values
    n = len(y)
    
    x = np.arange(n)
    poly = np.polyfit(x, y, 1)
    trend = np.polyval(poly, x)
    detrended_y = y - trend
    
    fft_vals = np.fft.rfft(detrended_y)
    frequencies = np.fft.rfftfreq(n)
    
    fft_vals_abs = np.abs(fft_vals)
    indices = np.argsort(fft_vals_abs)[::-1]
    
    fft_filtered = np.zeros_like(fft_vals)
    for i in range(num_harmonics):
        if i < len(indices):
            idx = indices[i]
            fft_filtered[idx] = fft_vals[idx]
            
    reconstructed_y = np.fft.irfft(fft_filtered, n=n)
    fitted_y = reconstructed_y + trend
    
    std_dev = np.std(y - fitted_y)
    
    forecast_x = np.arange(n + forecast_points)
    forecast_trend = np.polyval(poly, forecast_x)
    
    forecast_signal = np.zeros(n + forecast_points)
    for i in range(num_harmonics):
        if i < len(indices):
            idx = indices[i]
            amplitudes = np.abs(fft_vals[idx]) / n
            phases = np.angle(fft_vals[idx])
            freq = frequencies[idx]
            if idx == 0:
                forecast_signal += amplitudes
            else:
                forecast_signal += 2 * amplitudes * np.cos(2 * np.pi * freq * forecast_x + phases)
                
    forecast_y = forecast_signal + forecast_trend
    return fitted_y, forecast_y, std_dev

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fourier Anomaly Forecast</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
</head>
<body style="font-family: Arial; padding: 20px;">
    <h1>Fourier Fast-Transform Forecast</h1>
    <div style="background:#f5f5f5;padding:15px;border-radius:8px;margin-bottom:20px;word-break:break-all;">
        <strong>Metric Setup (from deployment):</strong> {{ metric_query }}
    </div>
    
    <div style="width: 90%; margin: auto;">
        <canvas id="fourierChart"></canvas>
    </div>
    <script>
        fetch('/api/data')
            .then(res => res.json())
            .then(data => {
                if(data.error) {
                    alert(data.error);
                    return;
                }
                const ctx = document.getElementById('fourierChart').getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.labels,
                        datasets: [
                            { label: 'Actual Data', data: data.actual, borderColor: 'black', fill: false, pointRadius: 0, borderWidth: 1 },
                            { label: 'Fitted/Forecast (yhat)', data: data.yhat, borderColor: 'blue', borderDash: [5, 5], fill: false, pointRadius: 0, borderWidth: 2 },
                            { label: 'Upper Bound', data: data.yhat_upper, borderColor: 'rgba(0, 255, 0, 0.3)', fill: false, pointRadius: 0, borderWidth: 1 },
                            { label: 'Lower Bound', data: data.yhat_lower, borderColor: 'rgba(0, 255, 0, 0.3)', fill: '-1', backgroundColor: 'rgba(0, 255, 0, 0.1)', pointRadius: 0, borderWidth: 1 }
                        ]
                    },
                    options: {
                        responsive: true,
                        interaction: { mode: 'index', intersect: false },
                        scales: { x: { type: 'time', time: { unit: 'day' } } }
                    }
                });
            });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, metric_query=METRIC_QUERY)

@app.route('/api/data')
def get_data():
    df = query_prometheus(START_TIME_STR, END_TIME_STR, step='15m')
    if df.empty:
        return jsonify({'error': 'No data retrieved'})
        
    forecast_points = 5 * 24 * 4 # 5 days at 15-min intervals
    fitted_y, forecast_y, std_dev = apply_fourier_transform(df, num_harmonics=10, forecast_points=forecast_points)
    
    last_timestamp = df['timestamp'].iloc[-1]
    forecast_timestamps = [last_timestamp + timedelta(minutes=15 * i) for i in range(1, forecast_points + 1)]
    
    all_timestamps = df['timestamp'].tolist() + forecast_timestamps
    actual_vals = df['y'].tolist() + [None] * forecast_points
    
    return jsonify({
        'labels': [t.strftime('%Y-%m-%dT%H:%M:%S') for t in all_timestamps],
        'actual': actual_vals,
        'yhat': forecast_y.tolist(),
        'yhat_upper': (forecast_y + 3 * std_dev).tolist(),
        'yhat_lower': (forecast_y - 3 * std_dev).tolist(),
    })

if __name__ == '__main__':
    # Listen on port 8080 exactly matching what K8s deployment expects
    app.run(host='0.0.0.0', port=8080, use_reloader=False)
