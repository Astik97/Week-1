import streamlit as st
st.set_page_config(page_title="🌊 Flood Risk Prediction Dashboard", layout="wide")

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Load trained models + artifacts
# ===============================
@st.cache_resource

def load_artifacts():
    model_data = joblib.load(r"B:\Videos\Flood_risk_prediction\flood_model.joblib")
    with open(r"B:\Videos\Flood_risk_prediction\metrics.json", "r") as f:
        metrics = json.load(f)
    return model_data, metrics

model_data, metrics = load_artifacts()
models = model_data["model"]
features = model_data["features"]
label_encoder = models["Label Encoder"]
scaler = models["Standard Scaler"]

# Load ML models
rfc = models["Random Forest Classifier"]
rfr = models["Random Forest Regressor"]
log_reg = models["Logistic Regression"]

# ===============================
# Streamlit Dashboard Layout
# ===============================
st.title("🌊 Flood Risk Prediction Dashboard")

st.markdown("""
Welcome to the *Flood Risk Prediction Dashboard* 🌊  

This app uses *Machine Learning Models* (Random Forest & Logistic Regression) trained on environmental and infrastructural features to estimate flood risk levels:  
- *Low*  
- *Moderate*  
- *High*  

👉 Use the sliders in the sidebar to input parameter values and generate predictions.
""")

# ===============================
# Sidebar Input Sliders
# ===============================
st.sidebar.header("📝 Input Feature Values")
user_data = {}
for feature in features:
    user_data[feature] = st.sidebar.slider(
        label=feature,
        min_value=0,
        max_value=10,
        value=5  # midpoint default value
    )

# Convert input to DataFrame
input_df = pd.DataFrame([user_data])

# ===============================
# Predictions
# ===============================
st.subheader("🔮 Flood Risk Prediction Results")
if st.button("Predict Flood Risk"):
    # Scale input for Logistic Regression
    scaled_input = scaler.transform(input_df)

    # Random Forest Classifier Prediction
    y_pred_cls = rfc.predict(input_df)[0]
    y_pred_cls_label = label_encoder.inverse_transform([y_pred_cls])[0]

    # Regression Prediction (Risk Score)
    y_pred_reg = rfr.predict(input_df)[0]

    # Logistic Regression Prediction
    y_pred_log = log_reg.predict(scaled_input)[0]
    y_pred_log_label = label_encoder.inverse_transform([y_pred_log])[0]

    # Display Results in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("🌲 Random Forest Prediction", y_pred_cls_label)
    col2.metric("📈 Random Forest Risk Score", f"{y_pred_reg:.2f}")
    col3.metric("⚖ Logistic Regression Prediction", y_pred_log_label)
else:
    st.info("Click the 'Predict Flood Risk' button to generate predictions.")

st.markdown("---")

# ===============================
# Model Performance (Confusion Matrix + Report)
# ===============================
st.subheader("📊 Model Evaluation Metrics (on Test Data)")

st.write("*Overall Accuracy:*", round(metrics["Accuracy"], 3))
st.write("*Classification Report:*")
st.json(metrics["Classification Report"])  # Expandable JSON format

st.markdown("---")

# ===============================
# Feature Importance Visualization
# ===============================
st.subheader("📌 Feature Importance (Random Forest Classifier)")
importances = pd.Series(rfc.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(6,6))
importances.plot(kind="barh", ax=ax, title="Feature Importance", color="teal")
st.pyplot(fig)

st.markdown("---")

# ===============================
# Input Data Summary
# ===============================
st.subheader("🧾 Your Input Data")
st.table(input_df)

st.markdown("---")

# ===============================
# Input Features Overview (clean bar chart)
# ===============================
st.subheader("📈 Input Features Overview")
values = input_df.iloc[0].astype(float)
# dynamic height so all feature labels are visible
height = max(4, 0.4 * len(values))
sorted_vals = values.sort_values()
fig, ax = plt.subplots(figsize=(8, height))
sns.set_style('whitegrid')
# Use matplotlib barh with an explicit color list to avoid passing ``palette`` to seaborn
colors = sns.color_palette('viridis', n_colors=len(sorted_vals))
positions = np.arange(len(sorted_vals))
ax.barh(positions, sorted_vals.values, color=colors)
ax.set_yticks(positions)
ax.set_yticklabels(sorted_vals.index)
ax.set_xlabel('Value')
ax.set_ylabel('Feature')
ax.set_title('Input Feature Values')
# annotate bars with values
x_max = sorted_vals.max() if len(sorted_vals) > 0 else 1
for i, v in enumerate(sorted_vals.values):
    ax.text(v + x_max * 0.01, i, f"{v:.2f}", va='center', fontsize=10)
fig.tight_layout()
st.pyplot(fig)