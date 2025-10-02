# Import the pandas library, our tool for working with dataframes
import pandas as pd

# Define the path to our training data
file_path = 'data/train_data.csv' # Or your correct filename

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# --- OLD CODE ABOVE ---
print("Successfully loaded the data. Here are the first 5 rows:")
print(df.head())


# --- NEW CODE BELOW ---

# 1. Get a summary of data types and non-null values
print("\n--- Data Types and Info ---")
print(df.info())

# 2. Check for missing values in each column
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 3. Get statistical summary for numerical columns
print("\n--- Statistical Summary ---")
print(df.describe())

# --- NEW CODE BELOW ---

# Step 3: Data Cleaning and Feature Engineering

# Drop the 'year' column as it has only one unique value
df = df.drop('year', axis=1)
print("\n--- Dropped the 'year' column ---")


# Convert month and day into a single datetime feature.
# Note: We are using a placeholder year (2008) since the original was dropped.
df['date'] = pd.to_datetime('2008-' + df['month'].astype(str) + '-' + df['day'].astype(str))
print("--- Created a 'date' column ---")


# We can also create features from the date, like the day of the week
df['day_of_week'] = df['date'].dt.dayofweek # Monday=0, Sunday=6
print("--- Created a 'day_of_week' column ---")


# Let's look at the first 5 rows again to see our new columns
print("\n--- DataFrame after initial cleaning and feature engineering ---")
print(df.head())