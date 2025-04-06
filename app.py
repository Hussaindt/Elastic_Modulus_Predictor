# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
MODEL_PATH = "xgboost_elastic_modulus_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Elastic Modulus Predictor", layout="centered")
st.title("🧱 Elastic Modulus Predictor for Concrete Beams")
st.markdown("Upload beam properties to predict Elastic Modulus at first crack.")

# Input method
input_method = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

feature_names = [
    'Concrete_Mix', 'Breadth', 'Depth', 'Length',
    'Moment_of_Inertia', 'Load_First_Crack', 'Deflection_First_Crack'
]

if input_method == "Manual Entry":
    st.subheader("Enter Beam Data")
    user_input = {
        'Concrete_Mix': st.number_input("Concrete Mix (N/mm²)", 10, 100, 30),
        'Breadth': st.number_input("Breadth (mm)", 50, 500, 150),
        'Depth': st.number_input("Depth (mm)", 50, 1000, 250),
        'Length': st.number_input("Length (mm)", 100, 5000, 1200),
        'Moment_of_Inertia': st.number_input("Moment of Inertia (mm⁴)", 100000, 1_000_000_000, 117187500),
        'Load_First_Crack': st.number_input("Load at First Crack (N)", 100, 100000, 4500),
        'Deflection_First_Crack': st.number_input("Deflection at First Crack (mm)", 0.01, 10.0, 0.6)
    }
    input_df = pd.DataFrame([user_input])

    if st.button("Predict Elastic Modulus"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Elastic Modulus: {prediction:.2f} N/mm²")

elif input_method == "Upload CSV":
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file with the required columns", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in feature_names):
            predictions = model.predict(df[feature_names])
            df['Predicted_Elastic_Modulus'] = predictions
            st.write("### Results:")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")
        else:
            st.error("CSV missing required columns. Please include all:")
            st.code(", ".join(feature_names))
