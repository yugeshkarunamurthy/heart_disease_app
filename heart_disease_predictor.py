import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('heart_disease_model.pkl')

# Streamlit UI
st.title("💓 Heart Disease Predictor")
st.markdown("Enter patient information to predict the presence of heart disease.")

# Input form
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
thalach = st.slider("Max Heart Rate Achieved (thalach)", 60, 210, 150)
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])

# Convert categorical values
sex_val = 1 if sex == "Male" else 0

# Create input DataFrame
input_data = pd.DataFrame([[age, sex_val, cp, thalach, oldpeak, ca]],
                          columns=['age', 'sex', 'cp', 'thalach', 'oldpeak', 'ca'])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 High risk of heart disease! (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Low risk of heart disease. (Confidence: {1 - probability:.2f})")
