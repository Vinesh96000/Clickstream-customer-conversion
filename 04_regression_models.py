import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the preprocessed training data
X_train_full = pd.read_csv('data/X_train.csv')
y_train_full = pd.read_csv('data/y_train.csv')

print("--- Data loaded successfully ---")

# Step 2: Prepare the data for regression
train_df = pd.concat([X_train_full, y_train_full], axis=1)
conversion_df = train_df[train_df['conversion'] == 1].copy()
print(f"--- Found {len(conversion_df)} converted sessions for regression task ---")

# Step 3: Define new Features (X_reg) and Target (y_reg)
y_reg = conversion_df['price']

# --- THE FIX IS HERE ---
# We drop the new 'price_2_2' column created by get_dummies, not the original 'price_2'.
X_reg = conversion_df.drop(['price', 'conversion', 'price_2_2'], axis=1)


# Step 4: Split the data for the regression model
X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
print("--- Regression data split into training and validation sets ---")

# Step 5: Build and Train the XGBRegressor Model
print("--- Training the XGBoost Regressor model ---")
xgb_regressor = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_regressor.fit(X_train_reg, y_train_reg)
print("--- Model training complete ---")

# Step 6: Evaluate the Regression Model
print("\n--- Evaluating the regression model ---")
y_pred_reg = xgb_regressor.predict(X_val_reg)
rmse = np.sqrt(mean_squared_error(y_val_reg, y_pred_reg))
r2 = r2_score(y_val_reg, y_pred_reg)

print(f"Validation Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Validation R-squared (R²): {r2:.2f}")