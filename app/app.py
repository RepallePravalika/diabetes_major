# app/app.py
import streamlit as st
import joblib
import numpy as np

st.title("🩺 Diabetes Detection System")

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
features = joblib.load('models/selected_features.pkl')

inputs = []
for f in features:
    val = st.number_input(f"Enter {f}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    X = scaler.transform([inputs])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    st.success(f"Prediction: {'Diabetic' if pred == 1 else 'Non-Diabetic'}")
    st.info(f"Probability: {prob:.2f}")
