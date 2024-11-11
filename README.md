# Dining Venue Guest Prediction Model

This project uses machine learning to predict the number of guests visiting a dining venue at any given time, helping managers optimize staffing by identifying peak hours.

## Data
| Adjusted Time | Meal     | Day of the Month | Concatenate | Actual Date | Day of the Week | # Guests | Table Type | Location |
|----------------|----------|------------------|-------------|-------------|-----------------|----------|------------|----------|
| 06:00 - 06:59  | Breakfast| 1                | 20240101    | 1/1/2024    | Monday          | 1        | Tab        | OLG      |
| 07:00 - 07:59  | Breakfast| 1                | 20240101    | 1/1/2024    | Monday          | 1        | Cashier    | OLG      |
| 08:00 - 08:59  | Breakfast| 1                | 20240101    | 1/1/2024    | Monday          | 16       | Cashier    | OLG      |
| 09:00 - 09:59  | Breakfast| 1                | 20240101    | 1/1/2024    | Monday          | 5        | Tab        | OLG      |
| 12:00 - 12:59  | Lunch    | 1                | 20240101    | 1/1/2024    | Monday          | 54       | Cashier    | OLG      |
| 14:00 - 14:59  | Lunch    | 1                | 20240101    | 1/1/2024    | Monday          | 10       | Cashier    | OLG      |
| 17:00 - 17:59  | Dinner   | 1                | 20240101    | 1/1/2024    | Monday          | 44       | Cashier    | OLG      |
| 19:00 - 19:59  | Dinner   | 1                | 20240101    | 1/1/2024    | Monday          | 7        | Cashier    | OLG      |
| ---            | Exclude  | 1                | 20240101    | 1/1/2024    | Monday          | 314      | ---        | OLG      |
| ...            | ...      | ...              | ...         | ...         | ...             | ...      | ...        | ...      | 

```python
import pandas as pd

# Load the dataset
df = pd.read_excel('Venue Utilization.xlsx')
```

### Columns:
- **Adjusted Time**: The time range (e.g., `06:00 - 06:59`) during which the guest count was recorded.
- **Meal**: The meal period (e.g., `Breakfast`, `Lunch`, `Dinner`).
- **Day of the Month**: The day of the month (e.g., `1` for January 1st).
- **Concatenate**: A numeric identifier concatenating the date for reference.
- **Actual Date**: The specific date in `MM/DD/YYYY` format.
- **Day of the Week**: The name of the day (e.g., `Monday`).
- **# Guests**: The number of guests recorded during that specific time slot.
- **Table Type**: The type of seating arrangement (e.g., `Cashier`, `Tab`).
- **Location**: The location or venue of the dining service (e.g., `OLG`, `Prom`, `Lola's`).

### Data Notes:
- Rows with **'---'** in the "Table Type" column represent **aggregate guest counts** for a specific "Adjusted Time" period or the **total number of guests for the entire day**. These rows should be **excluded** from the model as they do not represent individual guest counts for a specific meal period.
- Rows with **'Table Type' = 'Tab'** represent **employee meals** ordered at the Employee Kiosk. Can choose to include or remove these rows. 
- Guest counts are recorded for each unique combination of time, meal period, and table type.
- **Employee and resident to-go orders** placed through the Cubigo platform **cannot currently be identified** in the dataset.
- **Hours of operation** play a key role when predicting the number of guests.
  
The data provides insights into peak times for specific meals and helps identify patterns in guest behavior based on time of day, meal period, and venue location. This dataset is used for predicting guest attendance to optimize staffing decisions.

## Table of Contents
- [Model & Algorithms](#model--algorithms)
- [Testing](#testing)
- [End Product](#end-product)


## Model & Algorithms

1. Cleaning & Preparation:
 ```
 # Load required libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
 # Clean and preprocess data (similar to your code)
 df_cleaned = df[df["Meal"] != "Exclude"]
 df_cleaned = df_cleaned[df_cleaned["Table Type"].isin(["Cashier", "Table", "Quick Drinks"])]
 ```
    # Encode categorical variables
    label_encoders = {}
    for column in ['Meal', 'Day of the Week', 'Table Type', 'Location', 'Adjusted Time']:
      le = LabelEncoder()
      df_cleaned[column] = le.fit_transform(df_cleaned[column])
      label_encoders[column] = le
  ```
  # Prepare features and target variable
  X = df_cleaned[['Meal', 'Day of the Week', 'Adjusted Time', 'Location']]
  y = df_cleaned['# Guests']
  ```
2. Model & Algorithms:
  ```
  # Split the data into train and test sets 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
  
  # Train the model
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train_scaled, y_train)
  ```
   
## Testing

1. Testing:
 ```
 # Make predictions on the test set
 y_pred = model.predict(X_test_scaled)

 # Calculate metrics to evalute the model
 mae = mean_absolute_error(y_test, y_pred) 
 mse = mean_squared_error(y_test, y_pred)
 r2 = r2_score(y_test, y_pred)

 # Display model evaluation metrics
 print(mae)
 print(mse)
 print(r2)
 ```

## End Product

1. End Product:
 ```
 # Streamlit app
 st.title("Guest Prediction App")

 # Define ordered options
 meal_options = ["Breakfast", "Lunch", "Dinner"]
 day_of_week_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

 # Define mapping for locations
 location_mapping = {
   'OLG': "Oak Leaf Grille",
   'Prom': "Promenade",
   'Lola's': "Lola's"
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
   ```
