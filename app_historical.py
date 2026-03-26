import os, yaml, logging, requests
from datetime import datetime
from flask import Flask, render_template_string, jsonify
import pandas as pd
from prophet import Prophet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

CONFIG_PATH = os.environ.get('CONFIG_PATH', '/app/config.yaml')
PROMETHEUS_TOKEN = os.environ.get('PROMETHEUS_TOKEN', '')

def load_config():
    logger.info("Loading config")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config['prometheus']['url'], config['metrics']

def get_prometheus_headers():
    headers = {'Accept': 'application/json'}
    if PROMETHEUS_TOKEN: headers['Authorization'] = PROMETHEUS_TOKEN
    return headers

def query_prometheus(url, query, start_time, end_time, step=300):
    query_url = f"{url}/api/v1/query_range"
    params = {'query': query, 'start': start_time, 'end': end_time, 'step': step}
    try:
        response = requests.get(query_url, params=params, headers=get_prometheus_headers(), timeout=120)
        if response.status_code == 200 and response.json().get('data', {}).get('result'):
            df = pd.DataFrame(response.json()['data']['result'][0]['values'], columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'], unit='s')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            return df.dropna()
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
    return pd.DataFrame()

def train_prophet_model(df, periods, seasonality):
    model = Prophet(daily_seasonality=(seasonality == 'daily'), weekly_seasonality=(seasonality == 'weekly'), yearly_seasonality=False, interval_width=0.95)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='5min')
    forecast = model.predict(future)
    
    merged = df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
    merged['anomaly'] = (merged['y'] < merged['yhat_lower']) | (merged['y'] > merged['yhat_upper'])
    
    anomalies = merged[merged['anomaly'] == True]
    ano_list = [{ 'timestamp': row['ds'].strftime('%Y-%m-%dT%H:%M:%S'), 'actual': row['y'] } for _, row in anomalies.iterrows()]
    
    return forecast, ano_list

@app.route('/')
def index():
    try:
        _, metrics_config = load_config()
        metric_names = [m['name'] for m in metrics_config]
    except Exception as e:
        return f"Error loading config: {e}", 500

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
        <p>Select a metric below to compute and view its interactive forecast chart:</p>
        <ul>
        {% for name in metrics %}
            <li><a href="/chart/{{ name }}">{{ name }}</a></li>
        {% endfor %}
        </ul>
        <p><small>Note: Models are generated on-demand. Loading a chart may take a few moments.</small></p>
    </body>
    </html>
    """
    return render_template_string(html, metrics=metric_names)

@app.route('/chart/<metric_name>')
def chart_metric(metric_name):
    try:
        url, metrics_config = load_config()
    except Exception as e:
        return jsonify({'error': f'Config error: {e}'}), 500

    # Find metric config
    m_config = next((m for m in metrics_config if m['name'] == metric_name), None)
    if not m_config:
        return jsonify({'error': 'Metric not found in config'}), 404

    # Hardcoded dates as requested earlier
    start_time = "2026-02-16T12:00:00Z"
    end_time = "2026-03-04T05:21:16Z"
    
    logger.info(f"Dynamically querying and generating model for {metric_name}...")
    
    df = query_prometheus(url, m_config['query'], start_time, end_time)
    
    if df.empty:
         return f"<h1>Error</h1><p>No Prometheus data returned for metric <b>{metric_name}</b> between {start_time} and {end_time}.</p><a href='/'>Back</a>", 404

    forecast, anomalies = train_prophet_model(df, m_config.get('forecast_periods', 60), m_config.get('seasonality', 'daily'))
    
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
    app.run(host='0.0.0.0', port=8000, use_reloader=False)