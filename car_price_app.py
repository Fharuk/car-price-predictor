import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

# CONFIGURATION & UTILS
st.set_page_config(page_title="Car Price Estimator", layout="centered")
MODEL_FILENAME = "car_price_model_rf.pkl"
CURRENT_YEAR = datetime.now().year

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"Critical Error: '{MODEL_FILENAME}' not found.")
        return None
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_brands_from_model(model):
    """
    Dynamically finding all 'Brand_' features the model was trained on
    to populate the dropdown list automatically.
    """
    try:
        # Get all feature names the model expects
        model_features = model.feature_names_in_
        
        # Filter for columns that start with 'Brand_'
        brand_features = [f for f in model_features if f.startswith('Brand_')]
        
        # Clean up strings: 'Brand_Audi' -> 'Audi'
        brands = [f.replace('Brand_', '') for f in brand_features]
        return sorted(brands)
    except AttributeError:
        st.warning("Could not extract brands automatically. Using default list.")
        return ["Maruti", "Hyundai", "Honda", "Toyota", "Mercedes-Benz", "Volkswagen", "Ford", "Mahindra", "BMW", "Audi", "Tata"]

# UI & LOGIC
def main():
    st.title("🚗 Car Price Estimator (Enterprise Edition)")
    st.markdown("Enter vehicle details to estimate market value.")

    # 1. Load Model
    model = load_model()
    if model is None:
        st.stop()

    # 2. Get Dynamic Brand List
    available_brands = extract_brands_from_model(model)

    # 3. User Inputs
    st.sidebar.header("Vehicle Details")
    
    # Critical Missing Input: Brand
    brand = st.sidebar.selectbox("Car Brand", available_brands)
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", 1990, CURRENT_YEAR, 2015)
        km_driven = st.number_input("Kilometers", 0, 500000, 50000, step=1000)
        fuel_type = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        
    with col2:
        owner_type = st.selectbox("Owner", ["First", "Second", "Third", "Fourth & Above"])
        mileage = st.number_input("Mileage (kmpl)", 5.0, 50.0, 18.0)
        engine_cc = st.number_input("Engine CC", 600, 6000, 1500)
        power_bhp = st.number_input("Power (BHP)", 20.0, 800.0, 100.0)
        seats = st.number_input("Seats", 2, 10, 5)

    # 4. Preprocessing Adapter (The Fix)
    if st.button("Calculate Price"):
        try:
            # Step A: Create Raw DataFrame
            raw_data = pd.DataFrame({
                'Year': [year],
                'Kilometers_Driven': [km_driven],
                'Fuel_Type': [fuel_type],
                'Transmission': [transmission],
                'Owner_Type': [owner_type],
                'Mileage': [mileage],
                'Engine': [engine_cc],
                'Power': [power_bhp],
                'Seats': [seats],
                'Brand': [brand] # Added Brand
            })

            # Step B: Feature Engineering (BHP_per_CC)
            # We catch division by zero just in case
            raw_data['BHP_per_CC'] = raw_data['Power'] / raw_data['Engine']
            
            # Step C: Car Age
            raw_data['Car_Age'] = CURRENT_YEAR - raw_data['Year']

            # Step D: One-Hot Encoding (The Magic Step)
            # We convert our single row into OHE columns (Brand_Audi=1, Fuel_Diesel=1)
            data_encoded = pd.get_dummies(raw_data)

            # Step E: Alignment with Model Schema
            # This is the most critical line. It forces our data to match the model's expected columns EXACTLY.
            # 1. model.feature_names_in_ gives the exact list of columns the model wants.
            # 2. reindex creates missing columns (fill_value=0) and drops extra ones.
            model_columns = model.feature_names_in_
            data_final = data_encoded.reindex(columns=model_columns, fill_value=0)

            # Step F: Predict
            prediction = model.predict(data_final)[0]
            st.success(f"💰 Estimated Price: {prediction:.2f} Lakhs")
            
            # Debug info (Optional, helps verify alignment)
            with st.expander("Technical Debug Info"):
                st.write("Aligned Features:", data_final.columns.tolist())
                st.write("Data sent to model:", data_final)

        except Exception as e:
            st.error(f"Processing Error: {str(e)}")
            st.warning("Tip: This error usually means the input data structure doesn't match the training data.")

if __name__ == "__main__":
    main()
