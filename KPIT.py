#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datasets import load_dataset
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
# Load dataset
ds = load_dataset("infinite-dataset-hub/VehicleHealthMonitoring")
df = pd.DataFrame(ds['train'])
# Show dataset info
print("Dataset columns:", df.columns)
print("Sample data:\n", df.head())
# Preprocessing
# Convert categorical features to numerical
le = LabelEncoder()
df['diagnosis_encoded'] = le.fit_transform(df['diagnosis'])
# Prepare features and target
X = pd.get_dummies(df[['sensor_readings', 'error_codes', 'vehicle_age']]) 
y = df['diagnosis_encoded']
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))
# Streamlit Dashboard
st.title("🚗 Advanced Vehicle Diagnostic System")
st.write("Using real vehicle health monitoring data to predict issues")
# Input section
col1, col2 = st.columns(2)
with col1:
    sensor_value = st.slider("Sensor Reading", min_value=0.0, max_value=10.0, value=5.0)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=20, value=5)
with col2:
    error_code = st.selectbox("Error Code", options=df['error_codes'].unique())
    mileage = st.number_input("Mileage", min_value=0, value=50000)
# Predict button
if st.button("Run Diagnosis"):
    # Prepare input
    input_data = pd.DataFrame([{
        'sensor_readings': sensor_value,
        'error_codes': error_code,
        'vehicle_age': vehicle_age
    }])
    # One-hot encode
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max()
    # Display results
    st.subheader("Diagnosis Results")
    st.success(f"Predicted Issue: {le.inverse_transform([prediction])[0]}")
    st.info(f"Confidence: {probability:.2%}")
    # Show similar cases
    st.subheader("Similar Historical Cases")
    similar = df[df['diagnosis'] == le.inverse_transform([prediction])[0]].sample(3)
    st.dataframe(similar[['sensor_readings', 'error_codes', 'vehicle_age', 'diagnosis']])
# Add maintenance tips section
st.sidebar.title("Maintenance Tips")
st.sidebar.write("Based on your vehicle's profile:")
st.sidebar.write(f"- Recommended service interval: Every {max(3, 12 - vehicle_age)} months")
st.sidebar.write("- Check engine oil quality regularly")
st.sidebar.write("- Monitor tire pressure weekly")
# Add raw data explorer
if st.checkbox("Show Raw Data"):
    st.subheader("Vehicle Health Dataset")
    st.dataframe(df)


# In[ ]:




