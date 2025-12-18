import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pytz
import plotly.graph_objects as go
from typing import Literal
from pathlib import Path

st.set_page_config(page_title="Weather App", layout="centered")

icons_dir = Path(__file__).parent / "icons"

st.markdown(
    """
    <style>
        h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a {
            display: none !important;
        }
        
        .stMarkdown a[href^='#'] {
            display: none !important;
        }

        .stApp a[href^='#'] {
            display: none !important;
        }
        
        [data-testid="stImageContainer"]:nth-child(n+2) > img {
            height: 34px;
            width: 34px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_current_weather_for_city(city: str, API_KEY: str):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        return e


def analysis_time_series(
    df, city, date_col="timestamp", temp_col="temperature", window=30
):
    df = df[df["city"] == city].copy()

    if df.empty:
        st.warning(f"Нет данных для города: {city}")
        return None

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)

    df["rolling_mean"] = df[temp_col].rolling(window=window, center=True).mean()
    df["rolling_std"] = df[temp_col].rolling(window=window, center=True).std()

    df["upper_bound"] = df["rolling_mean"] + 2 * df["rolling_std"]
    df["lower_bound"] = df["rolling_mean"] - 2 * df["rolling_std"]

    df["is_anomaly"] = (df[temp_col] > df["upper_bound"]) | (
        df[temp_col] < df["lower_bound"]
    )
    anomalies = df[df["is_anomaly"]].copy()

    df["days_since_start"] = (df[date_col] - df[date_col].min()).dt.days
    X = df["days_since_start"].values
    y = df[temp_col].values

    valid = ~np.isnan(df["rolling_mean"])
    if valid.sum() < 2:
        st.error("Недостаточно данных для расчёта тренда.")
        return None

    X_valid, y_valid = X[valid], y[valid]
    A = np.vstack([X_valid, np.ones(len(X_valid))]).T
    slope, intercept = np.linalg.lstsq(A, y_valid, rcond=None)[0]
    df["trend"] = slope * df["days_since_start"] + intercept

    return {
        "df_enriched": df,
        "anomalies": anomalies,
        "trend_slope": slope,
        "trend_intercept": intercept,
    }


def graphic_time_series(
    df, city, anomalies, date_col="timestamp", temp_col="temperature", window=30
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    df["upper_bound"] = df["upper_bound"].fillna(method="ffill").fillna(method="bfill")
    df["lower_bound"] = df["lower_bound"].fillna(method="ffill").fillna(method="bfill")

    if not anomalies.empty:
        anomalies = anomalies.copy()
        anomalies[date_col] = pd.to_datetime(anomalies[date_col])
        anomalies = anomalies.sort_values(date_col).reset_index(drop=True)

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[temp_col],
            mode="lines",
            name="Temp",
            line=dict(color="lightgray", width=0.8),
            hovertemplate="%{y:.2f}°C<br>%{x}<extra></extra>",
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df["rolling_mean"],
            mode="lines",
            name=f"Moving average ({window} days)",
            line=dict(color="steelblue", width=2),
            hovertemplate="%{y:.2f}°C<br>%{x}<extra></extra>",
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df["trend"],
            mode="lines",
            name="Long term trend",
            line=dict(color="red", dash="dash", width=2),
            hovertemplate="%{y:.2f}°C<br>%{x}<extra></extra>",
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=df[date_col].tolist() + df[date_col][::-1].tolist(),
            y=df["upper_bound"].tolist() + df["lower_bound"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(255, 165, 0, 0.2)",
            line=dict(color="rgba(255, 165, 0, 0)"),
            name="±2σ",
            hoverinfo="skip",
        )
    )

    if not anomalies.empty:
        fig1.add_trace(
            go.Scatter(
                x=anomalies[date_col],
                y=anomalies[temp_col],
                mode="markers",
                name="Anomalies",
                marker=dict(color="red", size=4, opacity=0.7, symbol="circle-open"),
                hovertemplate="%{y:.2f}°C<br>%{x}<extra>Anomaly</extra>",
            )
        )

    fig1.update_layout(
        title=f"Analysis time series temp ({city})",
        xaxis_title="Date",
        yaxis_title="Temp (°C)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
    )

    st.plotly_chart(fig1, width="content")

    if not anomalies.empty and len(anomalies) <= 100:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df["rolling_mean"],
                mode="lines",
                name="Moving average",
                line=dict(color="steelblue"),
                hovertemplate="%{y:.2f}°C<br>%{x}<extra></extra>",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=anomalies[date_col],
                y=anomalies[temp_col],
                mode="markers",
                name="Anomalies",
                marker=dict(color="red", size=6, opacity=0.8),
                hovertemplate="%{y:.2f}°C<br>%{x}<extra>Anomaly</extra>",
            )
        )

        fig2.update_layout(
            title="Analysis values temp (Anomalies Highlighted)",
            xaxis_title="Date",
            yaxis_title="Temp (°C)",
            hovermode="x unified",
            height=300,
        )

        st.plotly_chart(fig2, width="content")


def plot_seasonal_profile(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    seasonal_unit: Literal["month", "dayofyear", "hour", "dayofweek"] = "month",
    title: str = "Seasonal Profile",
    show_std: bool = True,
    std_multiplier: float = 1.0,  # 1 → ±1σ, 2 → ±2σ
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if seasonal_unit == "month":
        df["season_group"] = df[date_col].dt.month
        x_label = "Month"
        x_tickvals = list(range(1, 13))
        x_ticktext = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    elif seasonal_unit == "dayofyear":
        df["season_group"] = df[date_col].dt.dayofyear
        x_label = "Day of Year"
        x_tickvals = None
        x_ticktext = None
    elif seasonal_unit == "hour":
        df["season_group"] = df[date_col].dt.hour
        x_label = "Hour of Day"
        x_tickvals = list(range(0, 24))
        x_ticktext = [f"{h}:00" for h in range(24)]
    elif seasonal_unit == "dayofweek":
        df["season_group"] = df[date_col].dt.dayofweek
        x_label = "Day of Week"
        x_tickvals = list(range(7))
        x_ticktext = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    else:
        raise ValueError("Unsupported seasonal_unit")

    agg = df.groupby("season_group")[value_col].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("season_group")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=agg["season_group"],
            y=agg["mean"],
            mode="lines+markers",
            name="Mean",
            line=dict(color="steelblue", width=2),
            marker=dict(size=4),
            hovertemplate="Mean: %{y:.2f}<br>" + x_label + ": %{x}<extra></extra>",
        )
    )

    if show_std and "std" in agg.columns:
        upper = agg["mean"] + std_multiplier * agg["std"]
        lower = agg["mean"] - std_multiplier * agg["std"]

        fig.add_trace(
            go.Scatter(
                x=agg["season_group"].tolist() + agg["season_group"][::-1].tolist(),
                y=upper.tolist() + lower[::-1].tolist(),
                fill="toself",
                fillcolor="rgba(173, 216, 230, 0.3)",
                line=dict(color="rgba(173, 216, 230, 0)"),
                name=f"±{std_multiplier}σ",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=value_col.capitalize(),
        hovermode="x unified",
        height=500,
        xaxis=dict(tickmode="array", tickvals=x_tickvals, ticktext=x_ticktext)
        if x_tickvals
        else {},
    )

    st.plotly_chart(fig, width="content")


def get_local_time_by_city(city_name):
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(city_name)

    if not location:
        return
        # raise ValueError(f"Город '{city_name}' не найден.")

    lat, lon = location.latitude, location.longitude

    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)

    if not timezone_str:
        return
        # raise ValueError(f"Не удалось определить часовой пояс для {city_name}.")

    tz = pytz.timezone(timezone_str)
    local_time = datetime.now(tz)

    return local_time


def prepare_current_date_to_history():
    pass


city_options = [
    "New York",
    "London",
    "Paris",
    "Tokyo",
    "Moscow",
    "Sydney",
    "Berlin",
    "Beijing",
    "Rio de Janeiro",
    "Dubai",
    "Los Angeles",
    "Singapore",
    "Mumbai",
    "Cairo",
    "Mexico City",
]

data = st.session_state.get("historical_data")

st.header("Weather App")

city = st.selectbox("Choose city", city_options, placeholder="Choose one option...")
api_key = st.session_state.get("api_key", "")

if city and api_key:
    res = get_current_weather_for_city(city, api_key)

    if res.get("cod") != 200:
        st.error(res.get("message"))
        st.stop()

    weather_main = res.get("weather")[0].get("main")
    description = res.get("weather")[0].get("description")

    main = res.get("main")
    temp = main.get("temp")
    feels_like = main.get("feels_like")
    temp_min = main.get("temp_min")
    temp_max = main.get("temp_max")

    pressure = main.get("pressure")

    humidity = main.get("humidity")

    sea_level = main.get("sea_level")
    grnd_level = main.get("grnd_level")

    visibility = res.get("visibility")

    country = res.get("sys").get("country")

    tz = timezone(timedelta(seconds=res["timezone"]))
    sunrise = datetime.fromtimestamp(res["sys"]["sunrise"], tz=tz).strftime("%H:%M")
    sunset = datetime.fromtimestamp(res["sys"]["sunset"], tz=tz).strftime("%H:%M")

    local_time = get_local_time_by_city(city).strftime("%H:%M")

    wind = res.get("wind")
    speed = wind.get("speed")
    gust = wind.get("gust")
    deg = wind.get("deg")

    image = (
        f"https://openweathermap.org/img/wn/{res.get('weather')[0].get('icon')}@2x.png"
    )

    st.markdown(
        f"##### Local time ({country},  {city}): {local_time}", text_alignment="center"
    )

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

    with col1:
        with st.container(
            border=True,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            st.subheader(city, text_alignment="center")
            st.header(f"{round(temp)} ℃", text_alignment="center")
            st.caption(
                f"Feels like: {str(round(feels_like))} ℃", text_alignment="center"
            )
            st.caption(
                f"H: {str(round(temp_max))} ℃ L: {str(round(temp_min))} ℃",
                text_alignment="center",
            )
            st.markdown(weather_main, text_alignment="center")
            if description.lower().strip() != weather_main.lower().strip():
                st.caption(description, text_alignment="center")
            st.image(image, channels="RGB")

    with col2:
        with st.container(
            border=True,
            width=100,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            st.image(f"{icons_dir}/sunrise.svg", caption="Sunrise", width=50)
            st.markdown(f"{sunrise} AM", text_alignment="center")

    with col3:
        with st.container(
            border=True,
            width=100,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            st.image(f"{icons_dir}/sunset.svg", caption="Sunset", width=50)
            st.markdown(f"{sunset} PM", text_alignment="center")

    with col4:
        with st.container(
            border=True,
            width=200,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            st.image(f"{icons_dir}/wind.svg", caption="Wind", width=50)
            st.markdown(f"speed - {speed}", text_alignment="center")
            st.markdown(f"gust - {gust}", text_alignment="center")
            st.markdown(f"deg - {deg}", text_alignment="center")

    with col2:
        with st.container(
            border=True,
            width=200,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            # st.image("icons/pressure.svg", caption="Pressure", width=50)
            st.caption("Pressure", text_alignment="center")
            st.markdown(pressure, text_alignment="center")

    with col3:
        with st.container(
            border=True,
            width=200,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            # st.image("icons/humidity.svg", caption="Humidity", width=50)
            st.caption("Humidity", text_alignment="center")
            st.markdown(humidity, text_alignment="center")

    with col2:
        with st.container(
            border=True,
            width=200,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            # st.image("icons/visibility.svg", caption="Visibility", width=50)
            st.caption("Visibility", text_alignment="center")
            st.markdown(visibility, text_alignment="center")

    with col3:
        with st.container(
            border=True,
            width=200,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            st.caption("sea_level", text_alignment="center")
            st.markdown(sea_level, text_alignment="center")

            st.caption("grnd_level", text_alignment="center")
            st.markdown(grnd_level, text_alignment="center")

    if len(data) > 0:
        st.divider()
        st.markdown("#### Graphics", text_alignment="center")

        with st.container(
            border=True,
        ):
            analysis = analysis_time_series(data, city)
            df_enriched = analysis.get("df_enriched")
            anomalies = analysis.get("anomalies")
            graphic_time_series(df_enriched, city, anomalies)

        with st.container(
            border=True,
        ):
            city_data = data[data["city"] == city]
            plot_seasonal_profile(
                df=city_data,
                date_col="timestamp",
                value_col="temperature",
                seasonal_unit="month",
                title=f"Monthly Temperature Profile ({city})",
                std_multiplier=2.0,
            )

elif api_key is None or api_key == "":
    st.switch_page("pages/settings.py")
