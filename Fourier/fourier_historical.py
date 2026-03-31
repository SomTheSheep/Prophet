from flask import Flask, jsonify, render_template_string
import requests
import urllib.parse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

PROM_URL = "https://prometheus.sitopflab03.otv-staging.com"
PROM_TOKEN = "your_token_here" # User should replace this
METRIC_QUERY = "histogram_quantile(0.90, sum(rate(istio_request_duration_milliseconds_bucket{app='ingress-gateway-otvpcse',request_url='/ias/v1/contentlicenses/widevine'}[10m])) by (le))"

def query_prometheus(start_time, end_time):
    headers = {'Authorization': f'Bearer {PROM_TOKEN}'}
    url = f"{PROM_URL}/api/v1/query_range"
    params = {
        'query': METRIC_QUERY,
        'start': start_time.isoformat() + "Z",
        'end': end_time.isoformat() + "Z",
        'step': '15m'
    }
    
    response = requests.get(url, headers=headers, params=params, verify=False)
    data = response.json()
    
    if data['status'] == 'success' and len(data['data']['result']) > 0:
        values = data['data']['result'][0]['values']
        df = pd.DataFrame(values, columns=['timestamp', 'y'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['y'] = df['y'].astype(float)
        # handle nan
        df['y'] = df['y'].fillna(method='ffill').fillna(0)
        return df
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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fourier Anomaly Forecast</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div style="width: 80%; margin: auto;">
        <canvas id="fourierChart"></canvas>
    </div>
    <script>
        fetch('/api/data')
            .then(res => res.json())
            .then(data => {
                const ctx = document.getElementById('fourierChart').getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.labels,
                        datasets: [
                            {
                                label: 'Actual Data',
                                data: data.actual,
                                borderColor: 'blue',
                                fill: false,
                                pointRadius: 0
                            },
                            {
                                label: 'Fitted/Forecast (yhat)',
                                data: data.yhat,
                                borderColor: 'red',
                                borderDash: [5, 5],
                                fill: false,
                                pointRadius: 0
                            },
                            {
                                label: 'Upper Bound',
                                data: data.yhat_upper,
                                borderColor: 'rgba(255, 99, 132, 0.2)',
                                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                fill: '+1',
                                pointRadius: 0
                            },
                            {
                                label: 'Lower Bound',
                                data: data.yhat_lower,
                                borderColor: 'rgba(255, 99, 132, 0.2)',
                                fill: false,
                                pointRadius: 0
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
                                type: 'category'
                            }
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
    start_time = datetime(2026, 2, 16, 12, 0, 0)
    end_time = datetime(2026, 3, 4, 5, 21, 16)
    
    df = query_prometheus(start_time, end_time)
    
    if df.empty:
        return jsonify({'error': 'No data retrieved'})
        
    forecast_points = 5 * 24 * 4 # 5 days of 15-min intervals
    fitted_y, forecast_y, std_dev = apply_fourier_transform(df, num_harmonics=10, forecast_points=forecast_points)
    
    last_timestamp = df['timestamp'].iloc[-1]
    forecast_timestamps = [last_timestamp + timedelta(minutes=15 * i) for i in range(1, forecast_points + 1)]
    
    all_timestamps = df['timestamp'].tolist() + forecast_timestamps
    actual_vals = df['y'].tolist() + [None] * forecast_points
    
    return jsonify({
        'labels': [t.strftime('%Y-%m-%d %H:%M:%S') for t in all_timestamps],
        'actual': actual_vals,
        'yhat': forecast_y.tolist(),
        'yhat_upper': (forecast_y + 3 * std_dev).tolist(),
        'yhat_lower': (forecast_y - 3 * std_dev).tolist(),
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
