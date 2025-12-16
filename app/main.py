import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pytz

st.set_page_config(page_title="Weather App", layout="centered")

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
    plt.figure(figsize=(14, 8))
    plt.plot(
        df[date_col],
        df[temp_col],
        label="Temp",
        color="lightgray",
        linewidth=0.8,
    )
    plt.plot(
        df[date_col],
        df["rolling_mean"],
        label=f"Moving average среднее ({window} days)",
        color="steelblue",
        linewidth=2,
    )
    plt.plot(
        df[date_col],
        df["trend"],
        label="Long term trend",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.fill_between(
        df[date_col],
        df["lower_bound"],
        df["upper_bound"],
        color="orange",
        alpha=0.2,
        label="±2σ",
    )
    if not anomalies.empty:
        plt.scatter(
            anomalies[date_col],
            anomalies[temp_col],
            color="red",
            label="Anomalies",
            zorder=5,
            s=30,
        )
    plt.title(f"Analysis time series temp ({city})")
    plt.xlabel("Date")
    plt.ylabel("Temp (°C)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    if not anomalies.empty and len(anomalies) <= 100:
        plt.figure(figsize=(12, 4))
        plt.plot(
            df[date_col],
            df["rolling_mean"],
            label="Moving average",
            color="steelblue",
        )
        plt.scatter(
            anomalies[date_col], anomalies[temp_col], color="red", label="Аномалии"
        )
        plt.title("Analysis values temp")
        plt.xlabel("Date")
        plt.ylabel("Temp (°C)")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()


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
api_key = st.session_state.get("api_key", "e9043dc6796e3ce6b7607e6d087f66af")

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
            st.image("icons/sunrise.svg", caption="Sunrise", width=50)
            st.markdown(f"{sunrise} AM", text_alignment="center")

    with col3:
        with st.container(
            border=True,
            width=100,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            st.image("icons/sunset.svg", caption="Sunset", width=50)
            st.markdown(f"{sunset} PM", text_alignment="center")

    with col4:
        with st.container(
            border=True,
            width=200,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            st.image("icons/wind.svg", caption="Wind", width=50)
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

    if data:
        st.markdown("##### Analysis time series temp", text_alignment="center")

        with st.container(
            border=True,
        ):
            analysis = analysis_time_series(data, city)
            df_enriched = analysis.get("df_enriched")
            anomalies = analysis.get("anomalies")
            graphic_time_series(df_enriched, city, anomalies)

elif api_key is None:
    st.switch_page("pages/settings.py")
