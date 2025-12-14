import streamlit as st


city_options = ["Moscow", "Berlin", "Paris", "Barcelona", "Madrid"]

st.header("Weather App")

city = st.multiselect("Choose city", city_options, placeholder="Choose one option...")

