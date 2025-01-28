# Python In-built packages
import os
from pathlib import Path
import PIL
import cv2
import joblib
import pandas as pd
# External packages
import streamlit as st
# import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Local Modules
# import settings
# import helper

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



# Load the model
model = joblib.load('heart_disease_model.pkl')