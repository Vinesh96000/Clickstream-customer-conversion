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

# --- Everything from the previous steps goes above this line ---
import matplotlib.pyplot as plt
import seaborn as sns

# Step 4: Visual Exploratory Data Analysis (EDA)

# Set the style for our plots
sns.set_style('whitegrid')

# --- Plot 1: Most Popular Product Categories ---
plt.figure(figsize=(10, 6))
sns.countplot(x='page1_main_category', data=df, order=df['page1_main_category'].value_counts().index)
plt.title('Distribution of Main Product Categories Viewed')
plt.xlabel('Main Category ID')
plt.ylabel('Number of Views')
# Create a folder to save plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')
plt.savefig('plots/category_distribution.png')
print("\n--- Saved plot for category distribution ---")


# --- Plot 2: User Traffic by Country ---
plt.figure(figsize=(12, 8))
# We'll plot the top 15 countries for readability
top_countries = df['country'].value_counts().nlargest(15).index
sns.countplot(y='country', data=df, order=top_countries, orient='h')
plt.title('Top 15 Countries by User Traffic')
plt.xlabel('Number of Sessions')
plt.ylabel('Country ID')
plt.savefig('plots/country_traffic.png')
print("--- Saved plot for country traffic ---")


# --- Plot 3: Price Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Distribution of Product Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.savefig('plots/price_distribution.png')
print("--- Saved plot for price distribution ---")



# Step 5: Create the Target Variable for Classification

# Find all session_ids that include a visit to page 5
sessions_with_purchase = df[df['page'] == 5]['session_id'].unique()

# Create the 'conversion' column. 
# Mark '1' if the session_id is in our purchase list, otherwise '0'.
df['conversion'] = df['session_id'].isin(sessions_with_purchase).astype(int)

print("\n--- Created the 'conversion' target variable ---")


# Let's check the balance of our new target variable
print("\n--- Conversion Distribution ---")
print(df['conversion'].value_counts())


# Let's see the first 5 rows with our new column
print("\n--- DataFrame with 'conversion' column ---")
print(df.head())

# --- Everything from the previous steps goes above this line ---
from sklearn.model_selection import train_test_split

# Step 6: Final Preprocessing and Splitting

# 1. Select Features (X) and Target (y)
# We drop columns that are identifiers, have been replaced, or are the target itself.
X = df.drop(['session_id', 'date', 'conversion'], axis=1)
y = df['conversion']

# 2. One-Hot Encode Categorical Features
# This will automatically convert 'page2_clothing_model' and any other non-numeric
# columns into a format the model can use. It will also handle our categorical
# numbers correctly if we specify them, but for now, we'll let it work on text.
X = pd.get_dummies(X, columns=['page2_clothing_model'], drop_first=True)

# We should also treat our "numeric" categories as categories.
# This avoids the model thinking country 2 is "better" than country 1.
categorical_cols = ['country', 'page1_main_category', 'colour', 'location', 'model_photography', 'price_2']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


print("\n--- Shape of data after One-Hot Encoding ---")
print(X.shape)



# 3. Split the data into training and VALIDATION sets
# We'll use an 80/20 split.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Data has been split into training and validation sets ---")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

# Step 7 (Part A): Save the processed data for modeling
X_train.to_csv('data/X_train.csv', index=False)
X_val.to_csv('data/X_val.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_val.to_csv('data/y_val.csv', index=False)

print("\n--- All processed data has been saved to the 'data' folder ---")