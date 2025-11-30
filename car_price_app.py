import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Car Price Estimator", layout="centered")

# Constants
MODEL_FILENAME = "car_price_model_rf.pkl"
CURRENT_YEAR = datetime.now().year

@st.cache_resource
def load_model():
    """
    Load the predictive model with error handling and caching.
    """
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"Critical Error: Model file '{MODEL_FILENAME}' not found.")
        return None
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_user_inputs():
    """
    Capture and organize user inputs from the sidebar or main area.
    """
    st.subheader("Vehicle Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Year of Manufacture", min_value=1990, max_value=CURRENT_YEAR, value=2015)
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])

    with col2:
        mileage = st.number_input("Mileage (kmpl or km/kg)", min_value=5.0, max_value=40.0, value=18.0, step=0.1)
        engine_cc = st.number_input("Engine CC", min_value=600, max_value=6000, value=1500, step=50)
        power_bhp = st.number_input("Power (BHP)", min_value=20.0, max_value=600.0, value=100.0, step=1.0)
        seats = st.number_input("Seats", min_value=2, max_value=14, value=5)
        tax = st.number_input("Tax (% or value)", min_value=0.0, max_value=50000.0, value=10.0, step=0.5)

    # Derived feature: Car Age
    car_age = CURRENT_YEAR - year

    # Construct DataFrame with columns matching the training schema
    input_data = pd.DataFrame({
        "Year": [year],
        "Kilometers_Driven": [km_driven],
        "Fuel_Type": [fuel_type],
        "Transmission": [transmission],
        "Owner_Type": [owner_type],
        "Mileage_num": [mileage],
        "Engine_num": [engine_cc],
        "Power_num": [power_bhp],
        "Seats": [seats],
        "Tax": [tax],
        "Car_Age": [car_age]
    })
    
    return input_data

def main():
    st.title("Car Price Estimator")
    st.markdown("Enter the vehicle details below to estimate the market price.")

    # Load Model
    model = load_model()
    if model is None:
        st.stop()

    # Get Inputs
    input_df = get_user_inputs()

    st.subheader("Estimation")
    if st.button("Calculate Price"):
        try:
            # Predict
            prediction = model.predict(input_df)[0]
            st.success(f"Estimated Market Value: {prediction:.2f} Lakhs")
            
            # Optional: Display input data for verification
            with st.expander("See Input Data Details"):
                st.dataframe(input_df)
                
        except ValueError as e:
            st.error(
                "Prediction Error: The model encountered unexpected input formats. "
                "Ensure categorical variables (Fuel, Transmission) match the training data encoding."
            )
            st.error(f"Details: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()