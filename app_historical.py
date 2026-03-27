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

def simulate_live_forward_walk(df, lookback_days=14, step_hours=6, forecast_days=5, seasonality='daily'):
    """
    Simulates the exact behavior of the live environment.
    Starts at day 14 of the dataset, trains, predicts future, saves a 6-hour slice, then repeats.
    """
    all_predictions = []
    
    min_date = df['ds'].min()
    max_date = df['ds'].max()
    
    # We need at least lookback_days to train the first model
    current_time = min_date + pd.Timedelta(days=lookback_days)
    
    # We will step forward by step_hours until we reach the end of what we want to simulate
    # In live, "app.py" trains every 6 hours and extracts T+5 for the next 6 hours.
    
    total_steps = int((max_date - current_time).total_seconds() / (step_hours * 3600))
    logger.info(f"Starting true live simulation. Lookback: {lookback_days}d. Step: {step_hours}h. Total steps to compute: {max(1, total_steps)}")

    while current_time < max_date:
        # 1. Isolate the dataset exactly up to current_time (the "past")
        train_df = df[(df['ds'] >= (current_time - pd.Timedelta(days=lookback_days))) & (df['ds'] <= current_time)]
        
        if len(train_df) > 10: # Ensure we have enough data to train
            # 2. Train the model blind to the future
            model = Prophet(daily_seasonality=(seasonality == 'daily'), weekly_seasonality=(seasonality == 'weekly'), yearly_seasonality=False, interval_width=0.95)
            # Suppress Prophet logs to keep console clean during hundreds of loops
            import logging as prophet_logging
            logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
            
            model.fit(train_df)
            
            # 3. Predict exactly far enough to cover our T+5 target
            # E.g. If current time is Day 14, we need to predict up to Day 14 + 5 days + 6 hours
            forecast_periods = int((forecast_days * 24 * 60) / 5) + int((step_hours * 60) / 5) 
            future = model.make_future_dataframe(periods=forecast_periods, freq='5min')
            forecast = model.predict(future)
            
            # 4. Extract the slice of predictions that the web server would have served.
            # In live, at `current_time + 1 hour`, the server looks for `current_time + 1 hour + 5 days`.
            # We want to extract the block of T+5 predictions that correspond to the next 6 hours of actuals.
            
            target_start = current_time + pd.Timedelta(days=forecast_days)
            target_end = target_start + pd.Timedelta(hours=step_hours)
            
            slice_df = forecast[(forecast['ds'] >= target_start) & (forecast['ds'] < target_end)][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            all_predictions.append(slice_df)
            
        # 5. Move clock forward by exactly 6 hours
        current_time += pd.Timedelta(hours=step_hours)
        
    if not all_predictions:
        return pd.DataFrame()
        
    return pd.concat(all_predictions).reset_index(drop=True)

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
                <h2>Future Forecasting</h2>
                <ul>
                {% for name in forecast_metrics %}
                    <li><a href="/chart/forecast/{{ name }}">{{ name }}</a></li>
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
    if chart_type not in ['anomaly', 'forecast', 'simulate']:
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

    if chart_type == 'simulate':
        # Generate the completely authentic walk-forward live simulation.
        # This uses the same 14-day lookback and 6-hour step as your live app.py
        shifted_forecast = simulate_live_forward_walk(df, lookback_days=14, step_hours=6, forecast_days=5, seasonality=m_config.get('seasonality', 'daily'))
        
        if shifted_forecast.empty:
            return f"<h1>Error</h1><p>Not enough historical data to simulate a 14-day lookback training window.</p><a href='/'>Back</a>", 404
            
        # The true "Live" predictions now start seamlessly around March 2nd (Feb 16 + 14d lookback) 
        # and stretch to ~March 9th. We want to combine the axes.
        
        # Merge actual timestamps and the future timestamps to build the single x-axis
        all_dates = sorted(list(set(df['ds'].tolist() + shifted_forecast['ds'].tolist())))
        timestamps = [pd.Timestamp(d).strftime('%Y-%m-%dT%H:%M:%S') for d in all_dates]
        
        actuals_dict = dict(zip(df['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), df['y']))
        actuals_mapped = [actuals_dict.get(t, "null") for t in timestamps]
        
        pred_dict = dict(zip(shifted_forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), shifted_forecast['yhat']))
        predicted_mapped = [pred_dict.get(t, "null") for t in timestamps]
        
        upper_dict = dict(zip(shifted_forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), shifted_forecast['yhat_upper']))
        upper_mapped = [upper_dict.get(t, "null") for t in timestamps]
        
        lower_dict = dict(zip(shifted_forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), shifted_forecast['yhat_lower']))
        lower_mapped = [lower_dict.get(t, "null") for t in timestamps]

        datasets_js = f"""
            {{label:'Actuals', data:[{",".join(map(str, actuals_mapped))}], borderColor:'black', borderWidth:1, fill:false, pointRadius:0}},
            {{label:'True Live Simulation (T+5)', data:[{",".join(map(str, predicted_mapped))}], borderColor:'blue', borderWidth:2, fill:false, pointRadius:0}},
            {{label:'Upper Bound', data:[{",".join(map(str, upper_mapped))}], borderColor:'rgba(0,255,0,0.3)', fill:false, pointRadius:0, borderWidth:1}},
            {{label:'Lower Bound', data:[{",".join(map(str, lower_mapped))}], borderColor:'rgba(0,255,0,0.3)', fill:'-1', backgroundColor:'rgba(0,255,0,0.1)', pointRadius:0, borderWidth:1}}
        """

    else:
        # For Anomaly and normal Forecast
        periods = m_config.get('forecast_periods', 60)
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

        elif chart_type == 'forecast':
            # Split into historical and future sections
            historical = forecast[forecast['ds'] <= max_actual_date]
            future = forecast[forecast['ds'] > max_actual_date]
            
            # Combine timestamps strictly ordered
            timestamps = forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
            
            # Actuals only exist up to the max date
            actuals_dict = dict(zip(actuals_df['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), actuals_df['y']))
            actuals_mapped = [actuals_dict.get(t, "null") for t in timestamps]
            
            # Future predictions start overlapping here
            future_pred_dict = dict(zip(future['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), future['yhat']))
            future_predicted_mapped = [future_pred_dict.get(t, "null") for t in timestamps]
            
            # Include future bounds
            future_upper_dict = dict(zip(future['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), future['yhat_upper']))
            future_upper_mapped = [future_upper_dict.get(t, "null") for t in timestamps]
            
            future_lower_dict = dict(zip(future['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'), future['yhat_lower']))
            future_lower_mapped = [future_lower_dict.get(t, "null") for t in timestamps]

            datasets_js = f"""
                {{label:'Historical Actuals', data:[{",".join(map(str, actuals_mapped))}], borderColor:'black', borderWidth:1, fill:false, pointRadius:0}},
                {{label:'Future Predicted', data:[{",".join(map(str, future_predicted_mapped))}], borderColor:'blue', borderWidth:2, fill:false, pointRadius:0}},
                {{label:'Future Upper Bound', data:[{",".join(map(str, future_upper_mapped))}], borderColor:'rgba(0,255,0,0.3)', fill:false, pointRadius:0, borderWidth:1}},
                {{label:'Future Lower Bound', data:[{",".join(map(str, future_lower_mapped))}], borderColor:'rgba(0,255,0,0.3)', fill:'-1', backgroundColor:'rgba(0,255,0,0.1)', pointRadius:0, borderWidth:1}}
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
        <strong>Type:</strong> {'True Rolling Walk-Forward' if chart_type == 'simulate' else 'Single Batch Training'}
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