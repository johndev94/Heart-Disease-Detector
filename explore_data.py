# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

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
# Split the dataset into features and target
x = data.drop('target', axis=1)
y = data['target']
print(y.value_counts())

# %%
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)   
# random_state is used to set the seed for random number generation.


# %%
# Train the model using random forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)


# %%
# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# %%
# Save the model
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved as heart_disease_model.pkl")


# %%
loaded_model = joblib.load('heart_disease_model.pkl')
sample_input = x_test.iloc[0].values.reshape(1, -1)  # Example input
sample_prediction = loaded_model.predict(sample_input)
print(f"Prediction for sample input: {sample_prediction}")
# %%
