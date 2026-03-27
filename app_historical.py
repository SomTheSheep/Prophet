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
    
    return forecast, ano_list, df

@app.route('/')
def index():
    try:
        _, metrics_config = load_config()
        metric_names = [m['name'] for m in metrics_config]
    except Exception as e:
        return f"Error loading config: {e}", 500

    excluded_forecast_metrics = ['rabbitmq_messages_ready_total', 'rabbitmq_active_consumers']
    forecast_metrics = [m for m in metric_names if m not in excluded_forecast_metrics]

    html = """
    <html>
    <head><title>Prophet Historical Forecasts</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .container { display: flex; gap: 50px; }
        .column { flex: 1; }
        ul { font-size: 16px; }
        li { margin-bottom: 8px; }
        a { text-decoration: none; color: #0066cc; font-weight: bold; }
        a:hover { text-decoration: underline; }
    </style>
    </head>
    <body>
        <h1>Prophet Historical Forecasts (Interactive)</h1>
        <p><small>Note: Models are generated on-demand. Loading a chart may take a few moments (usually ~5-15 seconds).</small></p>
        <div class="container">
            <div class="column">
                <h2>Anomaly Detection (Historical)</h2>
                <ul>
                {% for name in metric_names %}
                    <li><a href="/chart/anomaly/{{ name }}">{{ name }}</a></li>
                {% endfor %}
                </ul>
            </div>
            <div class="column">
                <h2>Continuous T+5 Forecast</h2>
                <ul>
                {% for name in forecast_metrics %}
                    <li><a href="/chart/simulate/{{ name }}">{{ name }}</a></li>
                {% endfor %}
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, metric_names=metric_names, forecast_metrics=forecast_metrics)

@app.route('/chart/<chart_type>/<metric_name>')
def chart_metric(chart_type, metric_name):
    if chart_type not in ['anomaly', 'simulate']:
        return "Invalid chart type", 400

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
    
    logger.info(f"Dynamically querying and generating model for {metric_name} ({chart_type})...")
    
    df = query_prometheus(url, m_config['query'], start_time, end_time)
    
    if df.empty:
         return f"<h1>Error</h1><p>No Prometheus data returned for metric <b>{metric_name}</b> between {start_time} and {end_time}.</p><a href='/'>Back</a>", 404

    periods = m_config.get('forecast_periods', 60)
    if chart_type == 'simulate':
        periods = 1440 # 5 days at 5-minute ticks (12 * 24 * 5 = 1440)

    forecast, anomalies, actuals_df = train_prophet_model(df, periods, m_config.get('seasonality', 'daily'))
    max_actual_date = actuals_df['ds'].max()
    
    if chart_type == 'anomaly':
        # Filter strictly up to the max historical date
        forecast = forecast[forecast['ds'] <= max_actual_date]
        
        timestamps = forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
        predicted = forecast['yhat'].tolist()
        lower = forecast['yhat_lower'].tolist()
        upper = forecast['yhat_upper'].tolist()
        
        # We need actual values for the labels
        actuals_dict = dict(zip(actuals_df['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), actuals_df['y']))
        actuals_mapped = [actuals_dict.get(t, "null") for t in timestamps]
        
        anomaly_times = [a['timestamp'] for a in anomalies]
        anomaly_values = [a['actual'] for a in anomalies]
        anomaly_data_js = "[" + ",".join([f"{{x: '{t}', y: {v}}}" for t, v in zip(anomaly_times, anomaly_values)]) + "]"

        datasets_js = f"""
            {{label:'Actuals', data:[{",".join(map(str, actuals_mapped))}], borderColor:'black', borderWidth:1, fill:false, pointRadius:0}},
            {{label:'Predicted', data:{predicted}, borderColor:'blue', borderWidth:2, fill:false, pointRadius:0}},
            {{label:'Upper', data:{upper}, borderColor:'rgba(0,255,0,0.3)', fill:false, pointRadius:0, borderWidth:1}},
            {{label:'Lower', data:{lower}, borderColor:'rgba(0,255,0,0.3)', fill:'-1', backgroundColor:'rgba(0,255,0,0.1)', pointRadius:0, borderWidth:1}},
            {{label:'Anomalies', data:{anomaly_data_js}, borderColor:'red', backgroundColor:'red', pointRadius:4, showLine:false}}
        """

    elif chart_type == 'simulate':
        # Start predictions 5 days after the beginning
        min_actual_date = actuals_df['ds'].min()
        t_plus_5_start = min_actual_date + pd.Timedelta(days=5)
        shifted_forecast = forecast[forecast['ds'] >= t_plus_5_start]
        
        # We need the full timestamp axis to support both actuals and the entire shifted forecast
        timestamps = forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
        
        # We need actual values mapped
        actuals_dict = dict(zip(actuals_df['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), actuals_df['y']))
        actuals_mapped = [actuals_dict.get(t, "null") for t in timestamps]
        
        # Mapped filtered predictions
        pred_dict = dict(zip(shifted_forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), shifted_forecast['yhat']))
        predicted_mapped = [pred_dict.get(t, "null") for t in timestamps]
        
        upper_dict = dict(zip(shifted_forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), shifted_forecast['yhat_upper']))
        upper_mapped = [upper_dict.get(t, "null") for t in timestamps]
        
        lower_dict = dict(zip(shifted_forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), shifted_forecast['yhat_lower']))
        lower_mapped = [lower_dict.get(t, "null") for t in timestamps]

        datasets_js = f"""
            {{label:'Actuals', data:[{",".join(map(str, actuals_mapped))}], borderColor:'black', borderWidth:1, fill:false, pointRadius:0}},
            {{label:'Predicted (Shifted T+5 Days)', data:[{",".join(map(str, predicted_mapped))}], borderColor:'blue', borderWidth:2, fill:false, pointRadius:0}},
            {{label:'Upper Bound', data:[{",".join(map(str, upper_mapped))}], borderColor:'rgba(0,255,0,0.3)', fill:false, pointRadius:0, borderWidth:1}},
            {{label:'Lower Bound', data:[{",".join(map(str, lower_mapped))}], borderColor:'rgba(0,255,0,0.3)', fill:'-1', backgroundColor:'rgba(0,255,0,0.1)', pointRadius:0, borderWidth:1}}
        """

    html = f"""<!DOCTYPE html>
    <html><head><title>Prophet {chart_type.title()}: {metric_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>body{{font-family:Arial;margin:20px}}.stats{{background:#f5f5f5;padding:15px;border-radius:8px;margin-bottom:20px}}</style>
    </head><body>
    <h2><a href="/"><- Back to Index</a></h2>
    <h1>Prophet {chart_type.title()}: {metric_name}</h1>
    <div class="stats">
        <strong>Chart Mode:</strong> {chart_type.title()} | 
        <strong>Total Data Points:</strong> {len(timestamps)} | 
        <strong>Forecast Bound:</strong> {periods * 5} minutes
    </div>
    <canvas id="chart" height="100"></canvas>
    <script>
    new Chart(document.getElementById('chart').getContext('2d'), {{
        type: 'line',
        data: {{
            labels: {timestamps},
            datasets: [
                {datasets_js}
            ]
        }},
        options: {{
            responsive:true,
            spanGaps: true,
            interaction: {{ mode: 'index', intersect: false }},
            scales: {{x:{{type:'time',time:{{unit:'day'}}}}}}
        }}
    }});
    </script></body></html>"""
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, use_reloader=False)