import os, yaml, logging, requests
from datetime import datetime
from flask import Flask, render_template_string, jsonify
import pandas as pd
from prophet import Prophet
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

CONFIG_PATH = os.environ.get('CONFIG_PATH', '/app/config.yaml')
PROMETHEUS_TOKEN = os.environ.get('PROMETHEUS_TOKEN', '')

metrics_data = {}
forecasts = {}
anomaly_results = {}
data_loaded = False

def load_config():
    global config, PROMETHEUS_URL, METRICS_CONFIG
    logger.info("Loading config")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    PROMETHEUS_URL = config['prometheus']['url']
    METRICS_CONFIG = config['metrics']

def get_prometheus_headers():
    headers = {'Accept': 'application/json'}
    if PROMETHEUS_TOKEN: headers['Authorization'] = PROMETHEUS_TOKEN
    return headers

def query_prometheus(query, start_time, end_time, step=300):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {'query': query, 'start': start_time, 'end': end_time, 'step': step}
    try:
        response = requests.get(url, params=params, headers=get_prometheus_headers(), timeout=120)
        if response.status_code == 200 and response.json()['data']['result']:
            df = pd.DataFrame(response.json()['data']['result'][0]['values'], columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'], unit='s')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            return df.dropna()
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
    return pd.DataFrame()

def train_prophet_and_store(name, df, periods, seasonality):
    model = Prophet(daily_seasonality=(seasonality == 'daily'), weekly_seasonality=(seasonality == 'weekly'), yearly_seasonality=False, interval_width=0.95)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='5min')
    forecast = model.predict(future)
    
    forecasts[name] = forecast
    
    merged = df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
    merged['anomaly'] = (merged['y'] < merged['yhat_lower']) | (merged['y'] > merged['yhat_upper'])
    
    anomalies = merged[merged['anomaly'] == True]
    ano_list = []
    for _, row in anomalies.iterrows():
        ano_list.append({
            'timestamp': row['ds'].strftime('%Y-%m-%dT%H:%M:%S'),
            'actual': row['y']
        })
    anomaly_results[name] = ano_list

def generate_all_data():
    global data_loaded
    logger.info("Starting background data generation...")
    try:
        load_config()
        start_time = "2026-02-16T12:00:00Z"
        end_time = "2026-03-04T05:21:16Z"
        
        for m in METRICS_CONFIG:
            name = m['name']
            query = m['query']
            periods = m.get('forecast_periods', 60)
            season = m.get('seasonality', 'daily')
            
            logger.info(f"Processing {name}...")
            df = query_prometheus(query, start_time, end_time)
            if not df.empty:
                metrics_data[name] = True
                train_prophet_and_store(name, df, periods, season)
        data_loaded = True
        logger.info("Background data generation complete!")
    except Exception as e:
        logger.error(f"Error generating data: {e}")

@app.route('/')
def index():
    html = """
    <html>
    <head><title>Prophet Historical Forecasts</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        ul { font-size: 18px; }
        li { margin-bottom: 10px; }
        a { text-decoration: none; color: #0066cc; font-weight: bold; }
        a:hover { text-decoration: underline; }
    </style>
    </head>
    <body>
        <h1>Prophet Historical Forecasts (Interactive)</h1>
        {% if not ready %}
            <p>Data is currently processing. Check pod logs and please refresh in a minute...</p>
        {% else %}
            <p>Select a metric below to view its interactive forecast chart:</p>
            <ul>
            {% for name in metrics %}
                <li><a href="/chart/{{ name }}">{{ name }}</a></li>
            {% endfor %}
            </ul>
        {% endif %}
    </body>
    </html>
    """
    return render_template_string(html, ready=data_loaded, metrics=list(metrics_data.keys()))

@app.route('/chart/<metric_name>')
def chart_metric(metric_name):
    if metric_name not in forecasts or metric_name not in anomaly_results:
        return jsonify({'error': 'Metric not found or still processing'}), 404

    forecast = forecasts[metric_name]
    anomalies = anomaly_results[metric_name]

    timestamps = forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
    predicted = forecast['yhat'].tolist()
    lower = forecast['yhat_lower'].tolist()
    upper = forecast['yhat_upper'].tolist()
    
    anomaly_times = [a['timestamp'] for a in anomalies]
    anomaly_values = [a['actual'] for a in anomalies]
    
    anomaly_data_js = "[" + ",".join([f"{{x: '{t}', y: {v}}}" for t, v in zip(anomaly_times, anomaly_values)]) + "]"

    html = f"""<!DOCTYPE html>
    <html><head><title>Prophet: {metric_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>body{{font-family:Arial;margin:20px}}.stats{{background:#f5f5f5;padding:15px;border-radius:8px;margin-bottom:20px}}</style>
    </head><body>
    <h2><a href="/"><- Back to Index</a></h2>
    <h1>Prophet Forecast: {metric_name}</h1>
    <div class="stats"><strong>Anomalies Detected:</strong> {len(anomalies)} | <strong>Total Data Points:</strong> {len(forecast)}</div>
    <canvas id="chart" height="100"></canvas>
    <script>
    new Chart(document.getElementById('chart').getContext('2d'), {{
        type: 'line',
        data: {{
            labels: {timestamps},
            datasets: [
                {{label:'Predicted',data:{predicted},borderColor:'blue',fill:false,pointRadius:0}},
                {{label:'Upper',data:{upper},borderColor:'rgba(0,255,0,0.3)',fill:false,pointRadius:0,borderWidth:1}},
                {{label:'Lower',data:{lower},borderColor:'rgba(0,255,0,0.3)',fill:'-1',backgroundColor:'rgba(0,255,0,0.1)',pointRadius:0,borderWidth:1}},
                {{label:'Anomalies',data:{anomaly_data_js},borderColor:'red',backgroundColor:'red',pointRadius:4,showLine:false}}
            ]
        }},
        options: {{
            responsive:true,
            interaction: {{ mode: 'index', intersect: false }},
            scales: {{x:{{type:'time',time:{{unit:'day'}}}}}}
        }}
    }});
    </script></body></html>"""
    return html

if __name__ == '__main__':
    # Run Generation in background thread so Gunicorn/Flask global states are respected
    threading.Thread(target=generate_all_data, daemon=True).start()
    app.run(host='0.0.0.0', port=8000, use_reloader=False)