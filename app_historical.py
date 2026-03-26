import os, yaml, logging, requests
from datetime import datetime
from flask import Flask, render_template_string
import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

CONFIG_PATH = os.environ.get('CONFIG_PATH', '/app/config.yaml')
PROMETHEUS_TOKEN = os.environ.get('PROMETHEUS_TOKEN', '')

# Ensure static folder exists for saving plots
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

metrics_plots = {}

def load_config():
    global config, PROMETHEUS_URL, METRICS_CONFIG
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

def train_prophet_and_plot(name, df, periods, seasonality):
    model = Prophet(daily_seasonality=(seasonality == 'daily'), weekly_seasonality=(seasonality == 'weekly'), yearly_seasonality=False, interval_width=0.95)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='5min')
    forecast = model.predict(future)
    
    # Generate anomaly plot (historical + bands)
    fig1 = model.plot(forecast)
    plt.title(f"{name} - Anomaly Bounds (Historical + Forecast)")
    anomaly_path = f"anomaly_{name}.png"
    fig1.savefig(os.path.join(STATIC_DIR, anomaly_path))
    plt.close(fig1)
    
    forecast_path = None
    if name not in ['rabbitmq_messages_ready_total', 'rabbitmq_active_consumers']:
        # Extract future forecast portion
        future_forecast = forecast[forecast['ds'] > df['ds'].max()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(future_forecast['ds'], future_forecast['yhat'], label='Forecast', color='orange')
        plt.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='blue', alpha=0.2)
        plt.title(f"{name} - Future Forecast")
        plt.legend()
        forecast_path = f"forecast_{name}.png"
        plt.savefig(os.path.join(STATIC_DIR, forecast_path))
        plt.close()
        
    return anomaly_path, forecast_path

def generate_all_plots():
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
            anomaly_path, forecast_path = train_prophet_and_plot(name, df, periods, season)
            metrics_plots[name] = {
                'anomaly': anomaly_path,
                'forecast': forecast_path
            }

@app.route('/')
def index():
    html = """
    <html>
    <head><title>Prophet Historical Forecasts</title></head>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .metric-container { margin-bottom: 40px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }
        img { max-width: 1000px; display: block; margin: 10px 0; }
    </style>
    <body>
        <h1>Prophet Historical Forecasts (Feb 16 - Mar 4)</h1>
        {% if not plots %}
            <p>Plots are generating, please refresh in a few minutes...</p>
        {% endif %}
        {% for name, paths in plots.items() %}
            <div class="metric-container">
                <h2>{{ name }}</h2>
                <h3>Anomaly Detection Plot (Actual vs Allowed Bounds)</h3>
                <img src="{{ url_for('static', filename=paths['anomaly']) }}" />
                {% if paths['forecast'] %}
                    <h3>Future Forecast Plot (Forecast only)</h3>
                    <img src="{{ url_for('static', filename=paths['forecast']) }}" />
                {% endif %}
            </div>
        {% endfor %}
    </body>
    </html>
    """
    return render_template_string(html, plots=metrics_plots)

if __name__ == '__main__':
    logger.info("Starting historical data processing...")
    generate_all_plots()
    logger.info("Completed processing. Starting web server...")
    app.run(host='0.0.0.0', port=8000)
