import streamlit as st
import joblib
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model_training import train_rf
from src.evaluation import evaluate_model
from src.utils import load_model

# Load the trained model
model = joblib.load("models/car_price_rf_model.pkl")

st.title("Car Price Prediction App")

# User inputs
year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, value=2018)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=7.5)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=35000)
owner = st.selectbox("Owner Count", [0,1,2,3])
fuel_type = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer","Individual"])
transmission = st.selectbox("Transmission", ["Manual","Automatic"])

# Prepare input
input_df = pd.DataFrame({
    'Year':[year],
    'Present_Price':[present_price],
    'Kms_Driven':[kms_driven],
    'Owner':[owner],
    'Fuel_Type':[fuel_type],
    'Seller_Type':[seller_type],
    'Transmission':[transmission]
})

input_df = pd.get_dummies(input_df, drop_first=True)
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Selling Price: {prediction[0]:.2f} lakhs")