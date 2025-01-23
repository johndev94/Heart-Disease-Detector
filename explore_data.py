# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# age: Age of the patient.
# sex: Gender (1 = male, 0 = female).
# cp: Chest pain type (categorical).
# trestbps: Resting blood pressure (in mm Hg).
# chol: Serum cholesterol in mg/dl.
# fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
# restecg: Resting electrocardiographic results (categorical).
# thalach: Maximum heart rate achieved.
# exang: Exercise-induced angina (1 = yes, 0 = no).
# oldpeak: ST depression induced by exercise relative to rest.
# slope: The slope of the peak exercise ST segment.
# ca: Number of major vessels (0-3) colored by fluoroscopy.
# thal: A categorical variable related to heart status.
# target: Target variable (1 = heart disease present, 0 = heart disease absent).

# %%
# Load the dataset
data = pd.read_csv('heart.csv') 

# %%
# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
data.isnull().sum()

# %%
