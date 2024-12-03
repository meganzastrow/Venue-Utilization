# Venue Utilization

import pandas as pd

# Load the dataset
df = pd.read_excel('Venue Utilization.xlsx')

# Load required libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import date, timedelta
import calendar
import plotly.graph_objects as go

# Clean and preprocess data
df_cleaned = df[df["Meal"] != "Exclude"]
df_cleaned = df_cleaned[df_cleaned["Table Type"].isin(["Cashier", "Table", "Quick Drinks"])]

df_cleaned.set_index("Actual Date", inplace=True)
df_cleaned['Month'] = df_cleaned.index.month

# Encode categorical variables
label_encoders = {}
for column in ['Meal', 'Day of the Week', 'Table Type', 'Location', 'Adjusted Time']:
  le = LabelEncoder()
  df_cleaned[column] = le.fit_transform(df_cleaned[column])
  label_encoders[column] = le


# Prepare features and target variable
X = df_cleaned[['Day of the Week', 'Adjusted Time', 'Location', 'Month', 'Day of the Month']]
y = df_cleaned['# Guests']

# Split the data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Train the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate metrics to evalute the model
mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model evaluation metrics
print(f"Mean Absolute Error: {mae: .2f}")
print(f"Mean Squared Error: {mse: .2f}")
print(f"R-Squared: {r2: .2f}")

# Streamlit app
st.title("Guest Prediction App")

# Define ordered options
#meal_options = ["Breakfast", "Lunch", "Dinner"]
day_of_week_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Define mapping for locations
location_mapping = {
  'OLG': "Oak Leaf Grille",
  'Prom': "Promenade",
  "Lola's": "Lola's"
 }

# Create a list of location options based on the mapping
location_options = list(location_mapping.values())

selected_date = st.date_input("Select a Date")
selected_month = selected_date.month
selected_day_of_the_month = selected_date.day
selected_day_of_week = selected_date.strftime("%A")

# Input meal, location, and time
#meal = st.selectbox("Select Meal Type", options=label_encoders['Meal'].classes_)
location = st.selectbox("Select Location", options=location_options)
time_of_day_input = st.multiselect("Select Time(s)", options=label_encoders['Adjusted Time'].classes_)

# When the user clicks the button
if st.button("Predict Number of Guests"):
  # Get the encoded value for the selected location
  location_encoded = next(key for key, value in location_mapping.items() if value == location)
   
  # Prepare input data for prediction
  total_predictions = 0

  for time in time_of_day_input:
    input_data = pd.DataFrame({
        #'Meal': [label_encoders['Meal'].transform([meal])[0]],
        'Day of the Week': [label_encoders['Day of the Week'].transform([selected_day_of_week])[0]],
        'Adjusted Time': [label_encoders['Adjusted Time'].transform([time])[0]],  
        'Location': [label_encoders['Location'].transform([location_encoded])[0]],
        'Month': [selected_month],
        'Day of the Month': [selected_day_of_the_month]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    total_predictions += int(prediction[0])

 # Display the result
  st.write(f"### Total Predicted Number of Guests: **{total_predictions}**")






# Import necessary libraries for the calendar
import plotly.figure_factory as ff

# Add a calendar heatmap after the prediction
st.header("Monthly Predicted Guest Counts")

# Select month and year
selected_year = st.selectbox("Select Year", options=[date.today().year, date.today().year + 1])
selected_month = st.selectbox("Select Month", options=list(calendar.month_name[1:]))

# Generate calendar heatmap data
if st.button("Generate Calendar"):
    # Get the total days in the selected month
    num_days = calendar.monthrange(selected_year, list(calendar.month_name).index(selected_month))[1]
    location_encoded = next(key for key, value in location_mapping.items() if value == location)
    
    # Get the first day of the month and align the weekday
    first_day = date(selected_year, list(calendar.month_name).index(selected_month), 1).weekday()  # First day of the month


    # Prepare data for predictions (sample, assuming you have your encoders, scaler, and model)
    dates = [date(selected_year, list(calendar.month_name).index(selected_month), day) for day in range(1, num_days + 1)]
    heatmap_data = []

    for day in dates:
        day_name = day.strftime("%A")
        daily_total = 0

        for time in time_of_day_input:  # Iterate through selected times
            input_data = pd.DataFrame({
                #'Meal': [label_encoders['Meal'].transform([meal])[0]],  # Single value encoding
                'Day of the Week': [label_encoders['Day of the Week'].transform([day_name])[0]],  # Single value encoding
                'Adjusted Time': [label_encoders['Adjusted Time'].transform([time])[0]],  # Ensure time is handled individually
                'Location': [label_encoders['Location'].transform([location_encoded])[0]],
                'Month': [day.month],
                'Day of the Month': [day.day]
            })
        
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)
        
            # Make prediction
            prediction = model.predict(input_data_scaled)
            daily_total += int(prediction[0])  # Add prediction to total

        #Store the total for each day
        heatmap_data.append({"Date": day, "Day": day.day, "Day of Week": day.weekday(), "Guests": daily_total})

    # Convert heatmap data to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create grid with the correct weekday alignment
    grid = np.full((6, 7), '', dtype=object)  # 6 rows (max weeks), 7 columns (days of the week)
    guest_counts = np.full((6, 7), np.nan)  # For heatmap values

    for index, row in heatmap_df.iterrows():
        week = (row['Date'].day + first_day - 1) // 7
        day_of_week = row['Date'].weekday()
        grid[week, day_of_week] = f"{row['Day']}\n({row['Guests']})"
        guest_counts[week, day_of_week] = row['Guests']

    
    # Create the heatmap with proper alignment
    fig = ff.create_annotated_heatmap(
        z=guest_counts,
        x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday'],
        y=[f"Week {i+1}" for i in range(6)],
        annotation_text=grid,
        colorscale='YlOrRd',
        showscale=True,
        zmin=0,
        zmax=np.nanmax(guest_counts)  # Optional for better scaling
    )

    # Update layout for better display
    fig.update_layout(
        title=f"Predicted Guest Counts with Day Numbers: {selected_month} {selected_year}",
        xaxis_title="Day of the Week",
        yaxis_title="Week of the Month",
        yaxis_autorange='reversed',
    )

    # Display the calendar heatmap
    st.plotly_chart(fig)

#python -m streamlit run test_utilization.py