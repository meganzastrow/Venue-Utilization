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
 
   
## Testing


## End Product


