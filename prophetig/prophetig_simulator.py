import os
import logging
import requests
import json
import time
import urllib3
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from prophet import Prophet
from sqlalchemy import create_engine


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ------------------------------------------------------------
# Environment / Config
# ------------------------------------------------------------

PROM_URL = os.environ.get(
    "FLT_PROM_URL",
    "https://prometheus.sitopflab03.otv-staging.com"
)

PROM_TOKEN = os.environ.get("FLT_PROM_ACCESS_TOKEN", "")

DB_URI = os.environ.get(
    "PROPHETIG_DB_URI",
    "postgresql://prophetig_user:prophetig_pass@prophetig-postgres-service:5432/prophetig_db"
)

METRICS_FILE = "/etc/prophetig/metrics.json"
SIMULATION_FILE = "/etc/prophetig/simulation.json"


def load_configs():
    metrics = {}
    simulation = {}

    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

    if os.path.exists(SIMULATION_FILE):
        with open(SIMULATION_FILE, "r") as f:
            simulation = json.load(f)

    return metrics, simulation


METRICS_CONFIG, SIM_CONFIG = load_configs()


TIME_START = SIM_CONFIG.get("LIVE_START", "2026-04-13T17:55:09Z")
TIME_END = SIM_CONFIG.get("LIVE_END", "2026-04-20T13:27:15Z")
TRAIN_WINDOW_DAYS = float(SIM_CONFIG.get("TRAIN_WINDOW_DAYS", 6.72))
RETRAIN_EVERY_HOURS = float(SIM_CONFIG.get("RETRAIN_EVERY_HOURS", 6))


# ------------------------------------------------------------
# Database
# ------------------------------------------------------------

engine = create_engine(DB_URI)


# ------------------------------------------------------------
# Prometheus helpers
# ------------------------------------------------------------

def get_prometheus_headers():
    headers = {"Accept": "application/json"}

    if PROM_TOKEN:
        headers["Authorization"] = (
            f"Bearer {PROM_TOKEN}"
            if not PROM_TOKEN.startswith("Bearer ")
            else PROM_TOKEN
        )

    return headers


def query_prometheus(query, start_time, end_time, step="5m"):
    if not query:
        return pd.DataFrame()

    query_url = f"{PROM_URL}/api/v1/query_range"

    params = {
        "query": query,
        "start": start_time,
        "end": end_time,
        "step": step
    }

    try:
        response = requests.get(
            query_url,
            params=params,
            headers=get_prometheus_headers(),
            verify=False,
            timeout=120
        )

        data = response.json()

        if (
            data.get("status") == "success"
            and len(data.get("data", {}).get("result", [])) > 0
        ):
            df = pd.DataFrame(
                data["data"]["result"][0]["values"],
                columns=["ds", "y"]
            )

            df["ds"] = pd.to_datetime(df["ds"], unit="s")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            df = df.dropna(subset=["ds", "y"])
            df = df.sort_values("ds")

            return df

        logger.warning(f"Prometheus returned no data for query: {query}")

    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")

    return pd.DataFrame()


# ------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------

def get_metric_bool(metric_cfg, key, default):
    value = metric_cfg.get(key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in ["true", "1", "yes", "y"]

    return bool(value)


def get_metric_float(metric_cfg, key, default):
    try:
        return float(metric_cfg.get(key, default))
    except Exception:
        return float(default)


def get_metric_int(metric_cfg, key, default):
    try:
        return int(metric_cfg.get(key, default))
    except Exception:
        return int(default)


# ------------------------------------------------------------
# Optimized Prophet model
# ------------------------------------------------------------

def apply_prophet_univariate(df_target, forecast_points, metric_name=None, metric_cfg=None):
    if metric_cfg is None:
        metric_cfg = {}

    df_merged = df_target.copy()
    df_merged = df_merged.sort_values("ds")
    df_merged["y"] = pd.to_numeric(df_merged["y"], errors="coerce")
    df_merged = df_merged.dropna(subset=["ds", "y"])

    if len(df_merged) < 20:
        logger.warning(f"[{metric_name}] Not enough data points: {len(df_merged)}")
        return pd.DataFrame(), None

    # Most infra metrics should never go below zero.
    non_negative = get_metric_bool(metric_cfg, "NON_NEGATIVE", True)

    if non_negative:
        df_merged["y"] = df_merged["y"].clip(lower=0)

    # Prophet params - safer defaults than previous aggressive ones.
    interval_width = get_metric_float(
        metric_cfg,
        "PROPHET_INTERVAL_WIDTH",
        os.environ.get("PROPHET_INTERVAL_WIDTH", 0.98)
    )

    changepoint_prior_scale = get_metric_float(
        metric_cfg,
        "PROPHET_CHANGEPOINT_PRIOR_SCALE",
        os.environ.get("PROPHET_CHANGEPOINT_PRIOR_SCALE", 0.08)
    )

    seasonality_prior_scale = get_metric_float(
        metric_cfg,
        "PROPHET_SEASONALITY_PRIOR_SCALE",
        os.environ.get("PROPHET_SEASONALITY_PRIOR_SCALE", 5.0)
    )

    changepoint_range = get_metric_float(
        metric_cfg,
        "PROPHET_CHANGEPOINT_RANGE",
        os.environ.get("PROPHET_CHANGEPOINT_RANGE", 0.90)
    )

    seasonality_mode = metric_cfg.get(
        "SEASONALITY_MODE",
        os.environ.get("PROPHET_SEASONALITY_MODE", "additive")
    )

    use_weekly = TRAIN_WINDOW_DAYS >= 6.5

    weekly_seasonality = get_metric_bool(
        metric_cfg,
        "WEEKLY_SEASONALITY",
        use_weekly
    )

    daily_seasonality = get_metric_bool(
        metric_cfg,
        "DAILY_SEASONALITY",
        True
    )

    logger.info(
        f"[{metric_name}] Prophet params: "
        f"interval_width={interval_width}, "
        f"changepoint_prior_scale={changepoint_prior_scale}, "
        f"seasonality_prior_scale={seasonality_prior_scale}, "
        f"changepoint_range={changepoint_range}, "
        f"seasonality_mode={seasonality_mode}, "
        f"weekly={weekly_seasonality}, "
        f"daily={daily_seasonality}"
    )

    model = Prophet(
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=False,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode=seasonality_mode,
        changepoint_range=changepoint_range
    )

    # Lower Fourier orders reduce overfitting.
    hourly_fourier = get_metric_int(metric_cfg, "HOURLY_FOURIER_ORDER", 5)
    six_hour_fourier = get_metric_int(metric_cfg, "SIX_HOUR_FOURIER_ORDER", 4)
    daily_spike_fourier = get_metric_int(metric_cfg, "DAILY_SPIKE_FOURIER_ORDER", 8)

    if hourly_fourier > 0:
        model.add_seasonality(
            name="hourly",
            period=1 / 24,
            fourier_order=hourly_fourier
        )

    if six_hour_fourier > 0:
        model.add_seasonality(
            name="6hour",
            period=6 / 24,
            fourier_order=six_hour_fourier
        )

    if daily_spike_fourier > 0:
        model.add_seasonality(
            name="daily_spike",
            period=1,
            fourier_order=daily_spike_fourier
        )

    try:
        model.fit(df_merged)
    except Exception as e:
        logger.error(f"[{metric_name}] Prophet fit failed: {e}")
        return pd.DataFrame(), None

    future = model.make_future_dataframe(
        periods=forecast_points,
        freq="5min"
    )

    try:
        forecast = model.predict(future)
    except Exception as e:
        logger.error(f"[{metric_name}] Prophet predict failed: {e}")
        return pd.DataFrame(), model

    # --------------------------------------------------------
    # Robust residual-based banding
    # --------------------------------------------------------

    fitted = forecast[forecast["ds"].isin(df_merged["ds"])][["ds", "yhat"]]

    residual_df = pd.merge(
        df_merged[["ds", "y"]],
        fitted,
        on="ds",
        how="inner"
    )

    if residual_df.empty:
        logger.warning(f"[{metric_name}] Empty residual dataframe; using Prophet native bands.")

        if non_negative:
            forecast["yhat"] = forecast["yhat"].clip(lower=0)
            forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
            forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

        return forecast, model

    residual_df["residual"] = residual_df["y"] - residual_df["yhat"]
    abs_residual = residual_df["residual"].abs()

    median_residual = residual_df["residual"].median()
    mad = np.median(np.abs(residual_df["residual"] - median_residual))

    robust_sigma = 1.4826 * mad

    q90_error = abs_residual.quantile(0.90)
    q95_error = abs_residual.quantile(0.95)
    q99_error = abs_residual.quantile(0.99)

    if np.isnan(robust_sigma) or robust_sigma <= 1e-9:
        robust_sigma = q95_error

    if np.isnan(robust_sigma) or robust_sigma <= 1e-9:
        robust_sigma = df_merged["y"].std()

    if np.isnan(robust_sigma) or robust_sigma <= 1e-9:
        robust_sigma = 0.01

    band_sigma_multiplier = get_metric_float(
        metric_cfg,
        "BAND_SIGMA_MULTIPLIER",
        os.environ.get("BAND_SIGMA_MULTIPLIER", 3.0)
    )

    minimum_band_width = get_metric_float(
        metric_cfg,
        "MINIMUM_BAND_WIDTH",
        os.environ.get("MINIMUM_BAND_WIDTH", 0.01)
    )

    # Main band width. This replaces std*6, p95*2, p99*2, max*0.95.
    band_width = max(
        robust_sigma * band_sigma_multiplier,
        q95_error,
        minimum_band_width
    )

    # Optional q99 contribution if needed for very spiky metrics.
    use_q99_error = get_metric_bool(
        metric_cfg,
        "USE_Q99_ERROR_IN_BAND",
        False
    )

    if use_q99_error:
        q99_weight = get_metric_float(metric_cfg, "Q99_ERROR_WEIGHT", 0.5)
        band_width = max(band_width, q99_error * q99_weight)

    y_min = df_merged["y"].min()
    y_max = df_merged["y"].max()
    y_median = df_merged["y"].median()
    y_mean = df_merged["y"].mean()
    y_std = df_merged["y"].std()
    y_q90 = df_merged["y"].quantile(0.90)
    y_q95 = df_merged["y"].quantile(0.95)
    y_q99 = df_merged["y"].quantile(0.99)

    if np.isnan(y_std):
        y_std = 0.0

    upper_quantile_multiplier = get_metric_float(
        metric_cfg,
        "UPPER_QUANTILE_MULTIPLIER",
        os.environ.get("UPPER_QUANTILE_MULTIPLIER", 1.35)
    )

    upper_sigma_multiplier = get_metric_float(
        metric_cfg,
        "UPPER_SIGMA_MULTIPLIER",
        os.environ.get("UPPER_SIGMA_MULTIPLIER", 5.0)
    )

    enable_upper_cap = get_metric_bool(
        metric_cfg,
        "ENABLE_UPPER_CAP",
        True
    )

    max_reasonable_upper = max(
        y_q99 * upper_quantile_multiplier,
        y_q95 * (upper_quantile_multiplier + 0.15),
        y_median + robust_sigma * upper_sigma_multiplier,
        y_mean + y_std * upper_sigma_multiplier,
        minimum_band_width
    )

    # Optional absolute cap, useful for CPU percentage metrics.
    absolute_upper_cap = metric_cfg.get("ABSOLUTE_UPPER_CAP")

    if absolute_upper_cap is not None:
        try:
            absolute_upper_cap = float(absolute_upper_cap)
            max_reasonable_upper = min(max_reasonable_upper, absolute_upper_cap)
        except Exception:
            logger.warning(f"[{metric_name}] Invalid ABSOLUTE_UPPER_CAP={absolute_upper_cap}")

    # Optional minimum upper cap so band does not become too tight.
    minimum_upper_cap = metric_cfg.get("MINIMUM_UPPER_CAP")

    if minimum_upper_cap is not None:
        try:
            minimum_upper_cap = float(minimum_upper_cap)
            max_reasonable_upper = max(max_reasonable_upper, minimum_upper_cap)
        except Exception:
            logger.warning(f"[{metric_name}] Invalid MINIMUM_UPPER_CAP={minimum_upper_cap}")

    logger.info(
        f"[{metric_name}] Band stats: "
        f"y_min={y_min:.4f}, y_median={y_median:.4f}, y_mean={y_mean:.4f}, "
        f"y_q90={y_q90:.4f}, y_q95={y_q95:.4f}, y_q99={y_q99:.4f}, y_max={y_max:.4f}, "
        f"robust_sigma={robust_sigma:.4f}, "
        f"q90_error={q90_error:.4f}, q95_error={q95_error:.4f}, q99_error={q99_error:.4f}, "
        f"band_width={band_width:.4f}, "
        f"max_reasonable_upper={max_reasonable_upper:.4f}"
    )

    # Replace Prophet native bands with controlled robust bands.
    forecast["yhat_upper"] = forecast["yhat"] + band_width
    forecast["yhat_lower"] = forecast["yhat"] - band_width

    if enable_upper_cap:
        forecast["yhat_upper"] = np.minimum(
            forecast["yhat_upper"],
            max_reasonable_upper
        )

    if non_negative:
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
        forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    # Ensure proper ordering.
    forecast["yhat_upper"] = np.maximum(
        forecast["yhat_upper"],
        forecast["yhat"]
    )

    forecast["yhat_lower"] = np.minimum(
        forecast["yhat_lower"],
        forecast["yhat"]
    )

    if non_negative:
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)

    return forecast, model


# ------------------------------------------------------------
# Historical backfill simulator
# ------------------------------------------------------------

def simulate_historical_data():
    fmt = "%Y-%m-%dT%H:%M:%SZ"

    virtual_now = datetime.strptime(TIME_START, fmt)
    end_dt = datetime.strptime(TIME_END, fmt)

    logger.info(
        f"Starting Prophetig optimized univariate backfill simulator "
        f"from {virtual_now} to {end_dt}"
    )

    while virtual_now < end_dt:
        logger.info(f"--- Processing Virtual Time Window: {virtual_now} ---")

        train_start = (virtual_now - timedelta(days=TRAIN_WINDOW_DAYS)).timestamp()
        train_end = virtual_now.timestamp()

        next_virtual_now = virtual_now + timedelta(hours=RETRAIN_EVERY_HOURS)

        if next_virtual_now > end_dt:
            next_virtual_now = end_dt

        predict_end = next_virtual_now.timestamp()

        # 5 min steps = 300 seconds.
        forecast_pts = int((predict_end - train_end) / 300)

        if forecast_pts <= 0:
            logger.warning(f"Skipping window because forecast_pts={forecast_pts}")
            virtual_now = next_virtual_now
            continue

        results_df_list = []

        for metric_name, queries in METRICS_CONFIG.items():
            if not isinstance(queries, dict):
                logger.warning(f"[{metric_name}] Invalid metric config. Skipping.")
                continue

            target_query = queries.get("TARGET_QUERY")

            if not target_query:
                logger.warning(f"[{metric_name}] Missing TARGET_QUERY. Skipping.")
                continue

            logger.info(
                f"[{metric_name}] Training {TRAIN_WINDOW_DAYS}d history, "
                f"predicting next {RETRAIN_EVERY_HOURS}h"
            )

            df_target_full = query_prometheus(
                target_query,
                train_start,
                predict_end,
                "5m"
            )

            if df_target_full.empty:
                logger.warning(f"[{metric_name}] No target data found.")
                continue

            # Hide future actuals from training.
            df_target_train = df_target_full[
                df_target_full["ds"] <= virtual_now
            ].copy()

            if df_target_train.empty:
                logger.warning(f"[{metric_name}] Empty training dataframe.")
                continue

            forecast, model = apply_prophet_univariate(
                df_target_train,
                forecast_points=forecast_pts,
                metric_name=metric_name,
                metric_cfg=queries
            )

            if forecast.empty:
                logger.warning(f"[{metric_name}] Empty forecast dataframe.")
                continue

            # Extract only next prediction block.
            forecast_future = forecast[
                (forecast["ds"] > virtual_now)
                & (forecast["ds"] <= next_virtual_now)
            ].copy()

            if forecast_future.empty:
                logger.warning(f"[{metric_name}] Empty future forecast block.")
                continue

            actuals_future = df_target_full[
                (df_target_full["ds"] > virtual_now)
                & (df_target_full["ds"] <= next_virtual_now)
            ].copy()

            actuals_future = actuals_future.rename(
                columns={"y": "actual_value"}
            )

            final_future = pd.merge(
                forecast_future,
                actuals_future[["ds", "actual_value"]],
                on="ds",
                how="left"
            )

            sql_df = pd.DataFrame({
                "timestamp": final_future["ds"],
                "metric_name": metric_name,
                "actual_value": final_future["actual_value"],
                "yhat": final_future["yhat"],
                "upper": final_future["yhat_upper"],
                "lower": final_future["yhat_lower"]
            })

            # Final SQL safety cleanup.
            non_negative = get_metric_bool(queries, "NON_NEGATIVE", True)

            sql_df["actual_value"] = pd.to_numeric(
                sql_df["actual_value"],
                errors="coerce"
            )

            sql_df["yhat"] = pd.to_numeric(
                sql_df["yhat"],
                errors="coerce"
            )

            sql_df["upper"] = pd.to_numeric(
                sql_df["upper"],
                errors="coerce"
            )

            sql_df["lower"] = pd.to_numeric(
                sql_df["lower"],
                errors="coerce"
            )

            if non_negative:
                sql_df["yhat"] = sql_df["yhat"].clip(lower=0)
                sql_df["upper"] = sql_df["upper"].clip(lower=0)
                sql_df["lower"] = sql_df["lower"].clip(lower=0)

            sql_df["upper"] = np.maximum(sql_df["upper"], sql_df["yhat"])
            sql_df["lower"] = np.minimum(sql_df["lower"], sql_df["yhat"])

            if non_negative:
                sql_df["lower"] = sql_df["lower"].clip(lower=0)

            results_df_list.append(sql_df)

        if results_df_list:
            master_df = pd.concat(results_df_list, ignore_index=True)

            logger.info(f"Writing {len(master_df)} rows to PostgreSQL")

            try:
                master_df.to_sql(
                    "prophetig_forecasts",
                    engine,
                    if_exists="append",
                    index=False
                )
            except Exception as e:
                logger.error(f"SQL Write Failed: {e}")

        virtual_now = next_virtual_now

    logger.info("✅ Historical Backfill Complete!")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    time.sleep(5)
    simulate_historical_data()