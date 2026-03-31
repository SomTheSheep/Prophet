from flask import Flask, jsonify, render_template_string
import os, yaml, logging, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

CONFIG_PATH = os.environ.get('CONFIG_PATH', 'config.yaml') # Defaults to local config.yaml
PROMETHEUS_TOKEN = "your_token_here" # Paste your actual token here

def load_config():
    logger.info("Loading config")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config['prometheus']['url'], config['metrics']

def get_prometheus_headers():
    headers = {'Accept': 'application/json'}
    if PROMETHEUS_TOKEN: headers['Authorization'] = f"Bearer {PROMETHEUS_TOKEN}" if not PROMETHEUS_TOKEN.startswith('Bearer ') else PROMETHEUS_TOKEN
    return headers

def query_prometheus(url, query, start_time, end_time, step='15m'):
    query_url = f"{url}/api/v1/query_range"
    params = {'query': query, 'start': start_time, 'end': end_time, 'step': step}
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
    
    # Detrend the data
    x = np.arange(n)
    poly = np.polyfit(x, y, 1)
    trend = np.polyval(poly, x)
    detrended_y = y - trend
    
    # FFT
    fft_vals = np.fft.rfft(detrended_y)
    frequencies = np.fft.rfftfreq(n)
    
    # Filter only top `num_harmonics` frequencies
    fft_vals_abs = np.abs(fft_vals)
    indices = np.argsort(fft_vals_abs)[::-1]
    
    # Reconstruct signal
    fft_filtered = np.zeros_like(fft_vals)
    for i in range(num_harmonics):
        if i < len(indices):
            idx = indices[i]
            fft_filtered[idx] = fft_vals[idx]
            
    reconstructed_y = np.fft.irfft(fft_filtered, n=n)
    fitted_y = reconstructed_y + trend
    
    # Standard deviation for bounds
    std_dev = np.std(y - fitted_y)
    
    # Forecast
    forecast_x = np.arange(n + forecast_points)
    forecast_trend = np.polyval(poly, forecast_x)
    
    # We reconstruct the signal for the forecast directly using the components
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

@app.route('/')
def index():
    try:
        _, metrics_config = load_config()
        metric_names = [m['name'] for m in metrics_config]
    except Exception as e:
        return f"Error loading config: {e}", 500

    html = """
    <html>
    <head><title>Fourier Historical Forecasts</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        ul { font-size: 16px; }
        li { margin-bottom: 8px; }
        a { text-decoration: none; color: #cc0066; font-weight: bold; }
        a:hover { text-decoration: underline; }
    </style>
    </head>
    <body>
        <h1>Fourier Fast-Transform Historical Forecasts</h1>
        <p><small>Note: Using identical configuration format as Prophet</small></p>
        <h2>Metrics Library</h2>
        <ul>
        {% for name in metric_names %}
            <li><a href="/chart/{{ name }}">{{ name }}</a></li>
        {% endfor %}
        </ul>
    </body>
    </html>
    """
    return render_template_string(html, metric_names=metric_names)

@app.route('/api/data/<metric_name>')
def get_data(metric_name):
    try:
        url, metrics_config = load_config()
    except Exception as e:
        return jsonify({'error': f'Config error: {e}'}), 500

    m_config = next((m for m in metrics_config if m['name'] == metric_name), None)
    if not m_config:
        return jsonify({'error': 'Metric not found in config'}), 404

    start_time = "2026-02-16T12:00:00Z"
    end_time = "2026-03-04T05:21:16Z"
    
    df = query_prometheus(url, m_config['query'], start_time, end_time, step='15m')
    
    if df.empty:
        return jsonify({'error': 'No data retrieved'})
        
    forecast_points = 5 * 24 * 4 # 5 days of 15-min intervals
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


@app.route('/chart/<metric_name>')
def chart_metric(metric_name):
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fourier Forecast: {{ metric_name }}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    </head>
    <body style="font-family: Arial;">
        <h2><a href="/" style="text-decoration:none;"><- Back to Index</a></h2>
        <h1>Fourier Forecast: {{ metric_name }}</h1>
        <div style="width: 90%; margin: auto;">
            <canvas id="fourierChart"></canvas>
        </div>
        <script>
            fetch('/api/data/{{ metric_name }}')
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
                                {
                                    label: 'Actual Data',
                                    data: data.actual,
                                    borderColor: 'black',
                                    fill: false,
                                    pointRadius: 0,
                                    borderWidth: 1
                                },
                                {
                                    label: 'Fitted/Forecast (yhat)',
                                    data: data.yhat,
                                    borderColor: 'blue',
                                    borderDash: [5, 5],
                                    fill: false,
                                    pointRadius: 0,
                                    borderWidth: 2
                                },
                                {
                                    label: 'Upper Bound',
                                    data: data.yhat_upper,
                                    borderColor: 'rgba(0, 255, 0, 0.3)',
                                    fill: false,
                                    pointRadius: 0,
                                    borderWidth: 1
                                },
                                {
                                    label: 'Lower Bound',
                                    data: data.yhat_lower,
                                    borderColor: 'rgba(0, 255, 0, 0.3)',
                                    fill: '-1',
                                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                                    pointRadius: 0,
                                    borderWidth: 1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            interaction: {
                                mode: 'index',
                                intersect: false,
                            },
                            scales: {
                                x: {
                                    type: 'time',
                                    time: { unit: 'day' }
                                }
                            }
                        }
                    });
                });
        </script>
    </body>
    </html>
    """
    return render_template_string(HTML_TEMPLATE, metric_name=metric_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, use_reloader=False)
