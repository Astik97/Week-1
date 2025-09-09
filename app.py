
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and features
model_dict = joblib.load(r"B:/Videos/Flood_risk_prediction/flood_model.joblib")
model = model_dict["model"]
features = model_dict["features"]

# Create a Streamlit app
st.title("Flood Risk Prediction Dashboard")

# Create input fields for features
st.header("Enter Feature Values")
feature_values = {}
for feature in features:
    feature_values[feature] = st.number_input(feature, min_value=0.0, max_value=10.0, value=5.0)

# Create a button to predict flood risk
if st.button("Predict Flood Risk"):
    # Create a dataframe with input feature values
    input_df = pd.DataFrame([feature_values])

    # Scale the input features
    scaler = model["Standard Scaler"]
    input_df_scaled = scaler.transform(input_df)

    # Make predictions using the model
    predictions = {}
    for model_name, model_instance in model.items():
        if model_name in ["Random Forest Classifier", "Logistic Regression"]:
            if model_name == "Logistic Regression":
                prediction = model_instance.predict(input_df_scaled)
            else:
                prediction = model_instance.predict(input_df)
            predictions[model_name] = prediction[0]
        elif model_name in ["Random Forest Regressor", "Linear Regression"]:
            if model_name == "Linear Regression":
                prediction = model_instance.predict(input_df_scaled)
            else:
                prediction = model_instance.predict(input_df)
            predictions[model_name] = prediction[0]

    # Display predictions
    st.header("Flood Risk Predictions")
    for model_name, prediction in predictions.items():
        st.write(f"{model_name}: {prediction}")

    # Map prediction to flood risk level
    def map_prediction_to_risk(prediction, model_name):
        if model_name in ["Random Forest Classifier", "Logistic Regression"]:
            if prediction == 0:
                return "Low"
            elif prediction == 1:
                return "High"
            else:
                return "Moderate"
        else:
            if prediction < 0.33:
                return "Low"
            elif prediction < 0.66:
                return "Moderate"
            else:
                return "High"

    # Display flood risk level
    st.header("Flood Risk Level")
    for model_name, prediction in predictions.items():
        risk_level = map_prediction_to_risk(prediction, model_name)
        st.write(f"{model_name}: {risk_level}")