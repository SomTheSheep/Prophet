import os, logging, requests, json
import numpy as np
import pandas as pd
from datetime import timedelta
from prophet import Prophet
from flask import Flask, jsonify, render_template_string, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

PROM_URL = os.environ.get('FLT_PROM_URL', 'https://prometheus.sitopflab03.otv-staging.com')
PROM_TOKEN = os.environ.get('FLT_PROM_ACCESS_TOKEN', '')
START_TIME_STR = os.environ.get('FLT_DATA_START_TIME', '2026-02-16T12:00:00Z')
END_TIME_STR = os.environ.get('FLT_DATA_END_TIME', '2026-03-04T05:21:16Z')

# Load the ConfigMap configurations
CONFIG_FILE = '/etc/fourier/endpoints.json'
ENDPOINTS_CONFIG = {}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        ENDPOINTS_CONFIG = json.load(f)
else:
    # Fallback for local testing without the mounted volume
    ENDPOINTS_CONFIG = {
        "widevine": {
            "TARGET_QUERY": "histogram_quantile(0.50, sum(rate(istio_request_duration_milliseconds_bucket{app='ingress-gateway-otvpcse',request_url='/ias/v1/contentlicenses/widevine'}[10m])) by (le))",
            "REGRESSOR_RPS": "sum(rate(istio_requests_total{app='ingress-gateway-otvpcse',request_url='/ias/v1/contentlicenses/widevine'}[10m]))"
        }
    }

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
    
    # Merge the historical fluctuating data into the future dataframe
    cols_to_merge = ['ds']
    if 'cpu' in df_merged.columns: cols_to_merge.append('cpu')
    if 'rps' in df_merged.columns: cols_to_merge.append('rps')
    if 'err5xx' in df_merged.columns: cols_to_merge.append('err5xx')
        
    if len(cols_to_merge) > 1:
        future = pd.merge(future, df_merged[cols_to_merge], on='ds', how='left')
        
    # Forward-fill only the newly created empty future rows
    future = future.ffill()

    forecast = m.predict(future)
    return df_merged, forecast, m

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multivariate Prophet Forecast</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        body { font-family: Arial; padding: 20px; }
        .config-box { background:#f5f5f5; padding:15px; border-radius:8px; margin-bottom:20px; }
        select { padding: 8px; font-size: 16px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Multivariate Prophet Anomaly Forecast</h1>
    
    <label for="endpointSelect"><strong>Select Endpoint:</strong></label>
    <select id="endpointSelect" onchange="fetchData()">
        {% for ep in endpoints %}
            <option value="{{ ep }}">{{ ep }}</option>
        {% endfor %}
    </select>

    <div class="config-box">
        <p><strong>Target:</strong> P50 Latency</p>
        <p><strong>Visibility:</strong> Plotting the mathematical impact (+/- milliseconds) of Traffic onto the latency baseline.</p>
        <p><strong>Forecast:</strong> 6 Hours forward</p>
    </div>
    
    <div style="width: 90%; margin: auto;">
        <canvas id="prophetChart"></canvas>
    </div>

    <script>
        let myChart = null;

        function fetchData() {
            const endpoint = document.getElementById('endpointSelect').value;
            
            fetch('/api/data?endpoint=' + encodeURIComponent(endpoint))
                .then(res => res.json())
                .then(data => {
                    if(data.error) {
                        alert(data.error);
                        return;
                    }
                    
                    const ctx = document.getElementById('prophetChart').getContext('2d');
                    if (myChart) {
                        myChart.destroy();
                    }

                    myChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.labels,
                            datasets: [
                                { label: 'Actual Latency (y)', data: data.actual, borderColor: 'black', fill: false, pointRadius: 0, borderWidth: 1 },
                                { label: 'Forecast Baseline (yhat)', data: data.yhat, borderColor: 'blue', borderDash: [5, 5], fill: false, pointRadius: 0, borderWidth: 2 },
                                { label: 'Upper Bound', data: data.yhat_upper, borderColor: 'rgba(255, 99, 132, 0.5)', fill: false, pointRadius: 0, borderWidth: 1 },
                                { label: 'Lower Bound', data: data.yhat_lower, borderColor: 'rgba(255, 99, 132, 0.5)', fill: '-1', backgroundColor: 'rgba(255, 99, 132, 0.2)', pointRadius: 0, borderWidth: 1 },
                                
                                // Regressor Impact plotted natively in milliseconds on the same Y-Axis
                                { label: 'RPS Latency Impact', data: data.impact_rps, borderColor: 'orange', fill: false, pointRadius: 0, borderWidth: 1 }
                            ]
                        },
                        options: {
                            responsive: true,
                            interaction: { mode: 'index', intersect: false },
                            scales: { 
                                x: { type: 'time', time: { unit: 'hour' } },
                                y: { type: 'linear', display: true, position: 'left', title: {display: true, text: 'Milliseconds (ms)'} }
                            }
                        }
                    });
                });
        }

        // Initial load
        window.onload = fetchData;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    endpoints = list(ENDPOINTS_CONFIG.keys())
    return render_template_string(HTML_TEMPLATE, endpoints=endpoints)

@app.route('/api/data')
def get_data():
    endpoint_key = request.args.get('endpoint', 'widevine')
    config = ENDPOINTS_CONFIG.get(endpoint_key, {})
    
    if not config:
        return jsonify({'error': 'Invalid endpoint selected'})

    df_target = query_prometheus(config.get('TARGET_QUERY'), START_TIME_STR, END_TIME_STR, step='10m')
    if df_target.empty:
        return jsonify({'error': 'No target data retrieved'})

    df_cpu = query_prometheus(config.get('REGRESSOR_CPU'), START_TIME_STR, END_TIME_STR, step='10m')
    df_rps = query_prometheus(config.get('REGRESSOR_RPS'), START_TIME_STR, END_TIME_STR, step='10m')
    df_5xx = query_prometheus(config.get('REGRESSOR_5XX'), START_TIME_STR, END_TIME_STR, step='10m')
        
    forecast_points = 36 # 6 hours
    df_merged, forecast, m = apply_prophet_multivariate(df_target, df_cpu, df_rps, df_5xx, forecast_points=forecast_points)
    
    actual_vals = df_merged['y'].tolist() + [None] * forecast_points

    # Extract the mathematical impact of the regressors
    impact_cpu = forecast['cpu'].tolist() if 'cpu' in forecast.columns else []
    impact_rps = forecast['rps'].tolist() if 'rps' in forecast.columns else []
    impact_err5xx = forecast['err5xx'].tolist() if 'err5xx' in forecast.columns else []

    return jsonify({
        'labels': [t.strftime('%Y-%m-%dT%H:%M:%S') for t in forecast['ds']],
        'actual': actual_vals,
        'yhat': forecast['yhat'].tolist(),
        'yhat_upper': forecast['yhat_upper'].tolist(),
        'yhat_lower': forecast['yhat_lower'].tolist(),
        'impact_cpu': impact_cpu,
        'impact_rps': impact_rps,
        'impact_err5xx': impact_err5xx
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, use_reloader=False)
