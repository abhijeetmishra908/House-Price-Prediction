import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved artifacts
xgb_model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# --- Page Config ---
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

# --- Custom CSS for background and design ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
    color: #000000;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: #f0f2f6;
}
h1 {
    color: #2c3e50;
    text-align: center;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- App Title ---
st.title("🏠 House Price Prediction App")
st.write("Fill in the details below and get an estimated house price instantly.")

# --- Input Fields ---
median_income = st.number_input("Median Income", min_value=0.0, value=3.0)
housing_median_age = st.number_input("Housing Median Age", min_value=0, value=20)
total_rooms = st.number_input("Total Rooms", min_value=0, value=2000)
total_bedrooms = st.number_input("Total Bedrooms", min_value=0, value=500)
population = st.number_input("Population", min_value=0, value=800)
households = st.number_input("Households", min_value=0, value=300)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)

ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

# --- Data Preparation ---
input_data = pd.DataFrame([[median_income, housing_median_age, total_rooms, total_bedrooms,
                            population, households, latitude, longitude, ocean_proximity]],
                          columns=["median_income", "housing_median_age", "total_rooms", "total_bedrooms",
                                   "population", "households", "latitude", "longitude", "ocean_proximity"])

# One-hot encode categorical column
input_data = pd.get_dummies(input_data)
# Align with training columns
input_data = input_data.reindex(columns=columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_data)

# --- Prediction Button ---
if st.button("🔮 Predict House Price"):
    prediction = xgb_model.predict(input_scaled)
    st.success(f"💰 Predicted House Price: ${prediction[0]:,.2f}")
    st.balloons()