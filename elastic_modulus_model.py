# elastic_modulus_model.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# --- Load and clean data ---
def load_and_prepare_data(filepath):
    data = pd.read_excel(filepath)
    data = data.drop(columns=[col for col in data.columns if 'Unnamed' in col])

    required_cols = [
        'Concrete Mix (N/mm^2)', 'Breadth mm', 'Depth mm', 'Length mm',
        'Moment of Inertia (I) mm^4', 'Load at first crack (N)',
        'Deflection at First Crack (mm)', 'Elastic modulus of Beam at first crack N/mm^2'
    ]
    data = data.dropna(subset=required_cols).copy()

    data['Deflection at First Crack (mm)'] = pd.to_numeric(
        data['Deflection at First Crack (mm)'], errors='coerce')
    data = data.dropna(subset=['Deflection at First Crack (mm)'])

    data.rename(columns={
        'Concrete Mix (N/mm^2)': 'Concrete_Mix',
        'Breadth mm': 'Breadth',
        'Depth mm': 'Depth',
        'Length mm': 'Length',
        'Moment of Inertia (I) mm^4': 'Moment_of_Inertia',
        'Load at first crack (N)': 'Load_First_Crack',
        'Deflection at First Crack (mm)': 'Deflection_First_Crack',
        'Elastic modulus of Beam at first crack N/mm^2': 'Elastic_Modulus'
    }, inplace=True)

    return data

# --- Train the model ---
def train_model(data):
    features = [
        'Concrete_Mix', 'Breadth', 'Depth', 'Length',
        'Moment_of_Inertia', 'Load_First_Crack', 'Deflection_First_Crack'
    ]
    target = 'Elastic_Modulus'

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Model Performance:")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

    return model

# --- Predict Elastic Modulus ---
def predict_elastic_modulus(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# --- Save the model ---
def save_model(model, filename="xgboost_elastic_modulus_model.pkl"):
    joblib.dump(model, filename)

# --- Main Script ---
if __name__ == "__main__":
    dataset_path = "Data_set.xlsx"
    data = load_and_prepare_data(dataset_path)
    model = train_model(data)
    save_model(model)

    example = pd.DataFrame.from_dict({
        'Concrete_Mix': [30],
        'Breadth': [150],
        'Depth': [250],
        'Length': [1200],
        'Moment_of_Inertia': [117187500],
        'Load_First_Crack': [4500],
        'Deflection_First_Crack': [0.6]
    })

    prediction = predict_elastic_modulus(model, example)
    print(f"\nPredicted Elastic Modulus: {prediction[0]:.2f} N/mm^2")
