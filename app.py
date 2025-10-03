import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data
patient_features = pd.read_csv("patient_features.csv")
daily_temp = pd.read_csv("daily_temp.csv")
forecast = pd.read_csv("forecast.csv")
clf = joblib.load("risk_classifier.pkl")

# App layout
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")
st.title("🧠 Patient Monitoring Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Risk Classification", "📈 Temperature Forecast", "🩺 Vitals Viewer"])

# Tab 1: Classification
with tab1:
    st.header("Patient Risk Classification")
    st.dataframe(patient_features[['PatientID', 'Age', 'AvgHeartRate', 'AvgTemp', 'HighRisk']])

    st.subheader("Predict Risk for New Patient")
    age = st.slider("Age", 20, 90, 50)
    hr = st.slider("Average Heart Rate", 60, 150, 85)
    temp = st.slider("Average Temperature", 35.0, 42.0, 37.0)
    days = st.slider("Active Days", 1, 365, 30)

    input_df = pd.DataFrame([[age, hr, temp, days]], columns=['Age', 'AvgHeartRate', 'AvgTemp', 'ActiveDays'])
    prediction = clf.predict(input_df)[0]
    st.write("🩺 Risk Prediction:", "🔴 High Risk" if prediction == 1 else "🟢 Low Risk")

# Tab 2: Forecast
with tab2:
    st.header("Temperature Forecast")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
    ax.set_title("30-Day Temperature Forecast")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

# Tab 3: Vitals Viewer
with tab3:
    st.header("Patient Vitals Summary")
    selected_id = st.selectbox("Select PatientID", patient_features['PatientID'])
    selected = patient_features[patient_features['PatientID'] == selected_id]
    st.write(selected.T)