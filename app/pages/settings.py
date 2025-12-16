import streamlit as st
import pandas as pd

st.set_page_config(page_title="Settings", layout="centered")
st.header("Application Settings")

st.subheader("1. Upload historical data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = {"city", "timestamp", "temperature", "season"}
        actual_columns = set(df.columns)

        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()
        
        try:
            pd.to_datetime(df["timestamp"], errors="raise")
        except Exception:
            st.error("The 'timestamp' column contains values that cannot be parsed as datetime.")
            st.stop()

        st.session_state["historical_data"] = df
        st.success("File uploaded and validated successfully!")

        with st.expander("Show data preview"):
            st.dataframe(df.head(), use_container_width=True)

    except UnicodeDecodeError:
        st.error("Invalid file encoding. Please ensure the file is UTF-8 encoded.")
    except pd.errors.EmptyDataError:
        st.error("File is empty.")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    if "historical_data" in st.session_state:
        st.info("Historical data already uploaded.")
    else:
        st.info("Please upload a CSV file with columns: city, timestamp, temperature, season.")

st.subheader("2. API Key")
api_key = st.text_input(
    "Enter your API key",
    value=st.session_state.get("api_key", ""),
    type="password",
    placeholder="Your secret key...",
    help="API key will be stored locally in this session"
)

if api_key:
    st.session_state["api_key"] = api_key
    st.success("API key saved!")
else:
    if "api_key" in st.session_state:
        st.success("API key already saved.")
    else:
        st.warning("API key not set. Some features may be unavailable.")

with st.expander("Advanced"):
    if st.button("Reset all settings"):
        st.session_state.pop("historical_data", None)
        st.session_state.pop("api_key", None)
        st.success("Settings reset!")
        st.rerun()
