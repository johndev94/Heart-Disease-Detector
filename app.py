# Python In-built packages
import os
from pathlib import Path
import PIL
import cv2
import joblib
import pandas as pd
# External packages
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import explore_data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler



# Setting page layout
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤︎",
    layout="wide",
    initial_sidebar_state="auto"
)

# Main page heading
st.title("Heart disease prediction")

# Sidebar
st.sidebar.header("Configuration")

age = st.sidebar.slider("Age", 20, 100, 50)

gender = st.sidebar.radio("Sex", ["Male", "Female"])
gender_numeric = 1 if gender == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)

chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)

fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
fbs_numeric = 1 if fbs == "True" else 0

restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
restecg_dict = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}

thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)

exang = st.sidebar.radio("Exercise-Induced Angina", ["Yes", "No"])
exang_numerica = 1 if exang == "Yes" else 0

oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 2.0)   

slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"]) 
slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)

thal = st.sidebar.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])
thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

if st.sidebar.button("Predict"):
    # Load the trained model
    model = joblib.load('heart_disease_model.pkl')
    
    st.success("Model loaded successfully")

    user_predict_data = {
    "age": age,
    "sex": gender_numeric,
    "cp": cp_dict[cp],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs_numeric,
    "restecg": restecg_dict[restecg],
    "thalach": thalach,
    "exang": exang_numerica,
    "oldpeak": oldpeak,
    "slope": slope_dict[slope],
    "ca": ca,
    "thal": thal_dict[thal]
}

    input_df_scaled = model.transform()

    print(user_predict_data)
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_predict_data])

    # Make prediction
    prediction = model.predict(input_df)

    
    

    # Display results
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("The model predicts a risk of heart disease.")
    else:
        st.success("The model predicts no significant risk of heart disease.")



user_predict_data = {
    "age": age,
    "sex": gender_numeric,  # Rename from "gender" to "sex"
    "cp": cp_dict[cp],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs_numeric,
    "restecg": restecg_dict[restecg],
    "thalach": thalach,
    "exang": exang_numerica,
    "oldpeak": oldpeak,
    "slope": slope_dict[slope],
    "ca": ca,
    "thal": thal_dict[thal]
}

model = joblib.load('heart_disease_model.pkl')
# Get model training feature names
if hasattr(model, "feature_names_in_"):
    print("Model expects features:", model.feature_names_in_)
else:
    print("Cannot retrieve feature names from the model.")

# Print user input features
print("User input features:", list(user_predict_data.keys()))

# st.markdown("""
# ### Dataset Information:
# - **age**: Age of the patient.
# - **sex**: Gender (1 = male, 0 = female).
# - **cp**: Chest pain type (categorical).
# - **trestbps**: Resting blood pressure (in mm Hg).
# - **chol**: Serum cholesterol in mg/dl.
# - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
# - **restecg**: Resting electrocardiographic results (categorical).
# - **thalach**: Maximum heart rate achieved.
# - **exang**: Exercise-induced angina (1 = yes, 0 = no).
# - **oldpeak**: ST depression induced by exercise relative to rest.
# - **slope**: The slope of the peak exercise ST segment.
# - **ca**: Number of major vessels (0-3) colored by fluoroscopy.
# - **thal**: A categorical variable related to heart status.
# - **target**: Target variable (1 = heart disease present, 0 = heart disease absent).
# """)




# Load the model
model = joblib.load('heart_disease_model.pkl')