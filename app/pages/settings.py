import streamlit as st


st.header("Setting up app")

file = st.file_uploader("Upload historical data")

api_key = st.text_input("Enter api key", placeholder="")
