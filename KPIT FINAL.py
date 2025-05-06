#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('engine_data.csv')


# In[3]:


data.info()


# In[4]:


data['Engine_power'] = data['Engine rpm'] * data['Lub oil pressure']


# In[5]:


data.describe()


# In[6]:


data['Temperature_difference'] = data['Coolant temp'] - data['lub oil temp']


# In[7]:


data.describe().T


# In[8]:


data.drop(['Engine_power'], axis=1, inplace=True)


# In[9]:


data.info()


# In[10]:


data.columns


# In[11]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = data.drop(['Engine Condition'], axis=1)
y = data['Engine Condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, max_features='sqrt',min_samples_leaf=5,min_samples_split=2,subsample=0.8)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)


# In[12]:


from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[13]:


get_ipython().run_line_magic('pip', 'install streamlit')


# In[14]:


import pickle
with open('hhmodel.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[15]:


probabilities = model.predict_proba(X_test)[:, 1]
threshold = 0.6  # Adjust as needed
maintenance_indices = []

for i in range(len(probabilities)):
    if probabilities[i] > threshold:
        maintenance_indices.append(i)
maintenance_ranges = []
start_index = maintenance_indices[0]
for i in range(1, len(maintenance_indices)):
    if maintenance_indices[i] != maintenance_indices[i-1] + 1:
        end_index = maintenance_indices[i-1]
        maintenance_ranges.append((start_index, end_index))
        start_index = maintenance_indices[i]
end_index = maintenance_indices[-1]
maintenance_ranges.append((start_index, end_index))
maintenance_date_ranges = []
for start, end in maintenance_ranges:
    start_date = X_test.index[start]
    end_date = X_test.index[end]
    maintenance_date_ranges.append((start_date, end_date))
for start, end in maintenance_date_ranges:
    print("Maintenance might be needed between", start, "and", end)


# In[16]:


thresholds = {
    'Engine rpm': 0.4,  # Lowered threshold assuming minor deviations might indicate potential issues.
    'Lub oil pressure': 0.7,  # Raised threshold to ensure early detection of lubrication system issues.
    'Fuel pressure': 0.6,  # Kept threshold unchanged assuming moderate deviations are indicative of potential problems.
    'Coolant pressure': 0.7,  # Raised threshold to detect cooling system issues more sensitively.
    'lub oil temp': 0.65,  # Slightly raised threshold to detect temperature anomalies earlier.
    'Coolant temp': 0.6,  # Kept threshold unchanged assuming moderate deviations might indicate potential issues.
    'Temperature_difference': 0.65  # Slightly raised threshold for early detection of temperature gradient anomalies.
}



# In[17]:


# Make probabilistic predictions
probabilities = model.predict_proba(X_test)[:, 1] 
print(probabilities) # Probabilities for class 1

# Identify indices where maintenance might be needed for each parameter
maintenance_indices = {}

for parameter, threshold in thresholds.items():
    parameter_indices = []
    for i in range(len(probabilities)):
        if probabilities[i] > threshold:
            parameter_indices.append(i)
    maintenance_indices[parameter] = parameter_indices

# Convert maintenance indices to a range of indices for each parameter
maintenance_ranges = {}

for parameter, indices in maintenance_indices.items():
    parameter_ranges = []
    start_index = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            end_index = indices[i-1]
            parameter_ranges.append((start_index, end_index))
            start_index = indices[i]
    end_index = indices[-1]
    parameter_ranges.append((start_index, end_index))
    maintenance_ranges[parameter] = parameter_ranges

# Convert index ranges to number of days for each parameter
maintenance_days = {}

for parameter, ranges in maintenance_ranges.items():
    parameter_days = []
    for start, end in ranges:
        num_days = end - start + 1
        parameter_days.append(num_days)
    maintenance_days[parameter] = parameter_days

# Display maintenance duration in days for each parameter
for parameter, days in maintenance_days.items():
    print("Maintenance might be needed for", parameter, "for", days, "days")


# In[18]:


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each parameter
roc_curves = {}
for parameter in thresholds.keys():
    fpr, tpr, _ = roc_curve(y_test, probabilities)  # Assuming y_test is your true labels
    roc_auc = auc(fpr, tpr)
    roc_curves[parameter] = (fpr, tpr, roc_auc)

# Plot ROC curve for each parameter
import matplotlib.pyplot as plt

plt.figure()
for parameter, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label='ROC curve for %s (AUC = %0.2f)' % (parameter, roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[19]:


# Display maintenance duration in weeks for each parameter
for parameter, days_list in maintenance_days.items():
    weeks_list = [int(days / 7) if days > 7 else 0 for days in days_list]
    print("Maintenance might be needed for", parameter, "for", weeks_list, "weeks")


# In[20]:


# Initialize a dictionary to store the maintenance classification for each parameter
maintenance_classification = {}

# Identify maintenance classification for each parameter based on the thresholds
for parameter, threshold in thresholds.items():
    parameter_classification = []
    for probability in probabilities:
        if probability > threshold:
            parameter_classification.append(1)  # Maintenance might be needed
        else:
            parameter_classification.append(0)  # No maintenance needed
    maintenance_classification[parameter] = parameter_classification

# Display maintenance classification for each parameter
for parameter, classification in maintenance_classification.items():
    print("Maintenance classification for", parameter, ":", classification)

# Initialize a list of dictionaries to store maintenance classification for each vehicle
vehicle_maintenance_classification = []

# Populate maintenance classification for each vehicle
for i in range(len(probabilities)):
    vehicle_classification = {}
    for parameter, classification in maintenance_classification.items():
        vehicle_classification[parameter] = classification[i]
    vehicle_maintenance_classification.append(vehicle_classification)

# Display maintenance classification for each vehicle
for i, vehicle_classification in enumerate(vehicle_maintenance_classification):
    print("Maintenance classification for Vehicle", i+1, ":", vehicle_classification)

# Initialize a list to store the overall maintenance classification for each vehicle
overall_maintenance = []

# Identify overall maintenance classification for each vehicle based on the maintenance classification for each parameter
for vehicle_classification in vehicle_maintenance_classification:
    if any(value == 1 for value in vehicle_classification.values()):
        overall_maintenance.append(1)  # Maintenance might be needed
    else:
        overall_maintenance.append(0)  # No maintenance needed

# Display overall maintenance classification for each vehicle
for i, maintenance_needed in enumerate(overall_maintenance):
    print("Overall maintenance classification for Vehicle", i+1, ":", maintenance_needed)


# In[21]:


# Dictionary containing binary values for each parameter
binary_values = {'Engine rpm': 1, 'Lub oil pressure': 0, 'Fuel pressure': 1, 'Coolant pressure': 0, 'lub oil temp': 0, 'Coolant temp': 1, 'Temperature_difference': 0}

# Dictionary containing weights for each parameter
weights = {'Engine rpm': 0.07, 'Lub oil pressure': 0.21, 'Fuel pressure': 0.14, 'Coolant pressure': 0.7, 'lub oil temp': 0.14, 'Coolant temp': 0.21, 'Temperature_difference': 0.7}

# Calculate the weighted sum
weighted_sum = sum(binary_values[param] * weights[param] for param in binary_values)

# Display the weighted sum
print("Weighted sum:", weighted_sum)


# In[23]:


import streamlit as st
import pickle
import numpy as np

# Load the trained model (replace 'model.pkl' with your actual file name)


# Define the customized ranges for each feature based on dataset statistics
custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
    'Temperature_difference': (-22.669427, 119.008526)
}

# Feature Descriptions
feature_descriptions = {
    'Engine rpm': 'Revolution per minute of the engine.',
    'Lub oil pressure': 'Pressure of the lubricating oil.',
    'Fuel pressure': 'Pressure of the fuel.',
    'Coolant pressure': 'Pressure of the coolant.',
    'lub oil temp': 'Temperature of the lubricating oil.',
    'Coolant temp': 'Temperature of the coolant.',
    'Temperature_difference': 'Temperature difference between components.'
}

# Engine Condition Prediction App
def main():
    st.title("Engine Condition Prediction")

    # Display feature descriptions
    st.sidebar.title("Feature Descriptions")
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}:** {description}")

    # Input widgets with customized ranges
    engine_rpm = st.slider("Engine RPM", min_value=float(custom_ranges['Engine rpm'][0]), 
                           max_value=float(custom_ranges['Engine rpm'][1]), 
                           value=float(custom_ranges['Engine rpm'][1] / 2))
    lub_oil_pressure = st.slider("Lub Oil Pressure", min_value=custom_ranges['Lub oil pressure'][0], 
                                 max_value=custom_ranges['Lub oil pressure'][1], 
                                 value=(custom_ranges['Lub oil pressure'][0] + custom_ranges['Lub oil pressure'][1]) / 2)
    fuel_pressure = st.slider("Fuel Pressure", min_value=custom_ranges['Fuel pressure'][0], 
                              max_value=custom_ranges['Fuel pressure'][1], 
                              value=(custom_ranges['Fuel pressure'][0] + custom_ranges['Fuel pressure'][1]) / 2)
    coolant_pressure = st.slider("Coolant Pressure", min_value=custom_ranges['Coolant pressure'][0], 
                                 max_value=custom_ranges['Coolant pressure'][1], 
                                 value=(custom_ranges['Coolant pressure'][0] + custom_ranges['Coolant pressure'][1]) / 2)
    lub_oil_temp = st.slider("Lub Oil Temperature", min_value=custom_ranges['lub oil temp'][0], 
                             max_value=custom_ranges['lub oil temp'][1], 
                             value=(custom_ranges['lub oil temp'][0] + custom_ranges['lub oil temp'][1]) / 2)
    coolant_temp = st.slider("Coolant Temperature", min_value=custom_ranges['Coolant temp'][0], 
                             max_value=custom_ranges['Coolant temp'][1], 
                             value=(custom_ranges['Coolant temp'][0] + custom_ranges['Coolant temp'][1]) / 2)
    temp_difference = st.slider("Temperature Difference", min_value=custom_ranges['Temperature_difference'][0], 
                                max_value=custom_ranges['Temperature_difference'][1], 
                                value=(custom_ranges['Temperature_difference'][0] + custom_ranges['Temperature_difference'][1]) / 2)

    # Predict button
    if st.button("Predict Engine Condition"):
        result, confidence = predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference)
        

        # Explanation
        if result == 0:
            st.info(f"The engine is predicted to be in a normal condition. As the  Confidence level of the machine is : {1.0 - confidence:.2%}")
        else:
            st.warning(f"Warning! Please investigate further,As the  Confidence level of the machine is : {1.0 - confidence:.2%}")

    # Reset button
    if st.button("Reset Values"):
        st.experimental_rerun()

# Function to predict engine condition
def predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference):
    input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[:, 1]  # For binary classification, adjust as needed
    return prediction[0], confidence[0]

if __name__ == "__main__":
    main()


# In[ ]:




