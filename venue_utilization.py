import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load your data
df = pd.read_excel('Venue Utilization.xlsx')

# Clean and preprocess data (similar to your code)
df_cleaned = df[df["Meal"] != "Exclude"]
df_cleaned = df_cleaned[df_cleaned["Table Type"].isin(["Cashier", "Table", "Quick Drinks"])]

# Encode categorical variables
label_encoders = {}
for column in ['Meal', 'Day of the Week', 'Table Type', 'Location', 'Adjusted Time']:
    le = LabelEncoder()
    df_cleaned[column] = le.fit_transform(df_cleaned[column])
    label_encoders[column] = le

# Prepare features and target variable
X = df_cleaned[['Meal', 'Day of the Week', 'Adjusted Time', 'Location']]
y = df_cleaned['# Guests']

# Train the model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Streamlit app
st.title("Guest Prediction App")

# Define ordered options
meal_options = ["Breakfast", "Lunch", "Dinner"]
day_of_week_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Define mapping for locations
location_mapping = {
    'OLG': "Oak Leaf Grille",
    'Prom': "Promenade",
    "Lola's": "Lola's"
}

# Create a list of location options based on the mapping
location_options = list(location_mapping.values())

# Drop-downs for user input
meal = st.selectbox("Select Meal Type", options=meal_options)
day_of_week = st.selectbox("Select Day of the Week", options=day_of_week_options)
time_of_day_input = st.selectbox("Select Time", options=label_encoders['Adjusted Time'].classes_)
location = st.selectbox("Select Location", options=location_options)

# When the user clicks the button
if st.button("Predict Number of Guests"):
    # Get the encoded value for the selected location
    location_encoded = next(key for key, value in location_mapping.items() if value == location)
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Meal': [label_encoders['Meal'].transform([meal])[0]],
        'Day of the Week': [label_encoders['Day of the Week'].transform([day_of_week])[0]],
        'Adjusted Time': [label_encoders['Adjusted Time'].transform([time_of_day_input])[0]],  
        'Location': [label_encoders['Location'].transform([location_encoded])[0]]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    st.write(f'Predicted number of guests: {int(prediction[0])}')

    #Run this code in the terminal
    #python -m streamlit run venue_utilization.py
