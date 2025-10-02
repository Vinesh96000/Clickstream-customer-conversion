import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor

print("--- Starting artifact saving process ---")

# --- 1. Save the Classification Model ---
print("Training and saving the final classification model...")
# Load data
X_train_full = pd.read_csv('data/X_train.csv')
y_train_full = pd.read_csv('data/y_train.csv').values.ravel()

# Define and train the final XGBoost pipeline
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])
xgb_pipeline.fit(X_train_full, y_train_full)

# Save the pipeline object
joblib.dump(xgb_pipeline, 'classification_model.joblib')
# We also need to save the columns, so we can re-create the input in the app
joblib.dump(X_train_full.columns, 'classification_model_columns.joblib')

print("Classification model saved successfully.")


# --- 2. Save the Regression Model ---
print("Training and saving the final regression model...")
# Prepare data for regression
train_df = pd.concat([X_train_full, pd.DataFrame(y_train_full, columns=['conversion'])], axis=1)
conversion_df = train_df[train_df['conversion'] == 1].copy()
y_reg = conversion_df['price']
X_reg = conversion_df.drop(['price', 'conversion', 'price_2_2'], axis=1)

# Define and train the final XGBoost Regressor
xgb_regressor = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_regressor.fit(X_reg, y_reg)

# Save the model and its columns
joblib.dump(xgb_regressor, 'regression_model.joblib')
joblib.dump(X_reg.columns, 'regression_model_columns.joblib')

print("Regression model saved successfully.")


# --- 3. Save the Clustering Data ---
print("Training and saving the final clustering model and data...")
# Load original data
df = pd.read_csv('data/train_data.csv')

# Create session-level features
session_df = df.groupby('session_id').agg(
    total_clicks=('order', 'max'),
    avg_price=('price', 'mean'),
    distinct_categories=('page1_main_category', 'nunique'),
    sale_page_visited=('page1_main_category', lambda x: 1 if 4 in x.values else 0)
).reset_index()

# Scale features and train KMeans
features_for_clustering = session_df.drop('session_id', axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(scaled_features)

# Add cluster labels and save the final dataframe
session_df['cluster'] = kmeans.labels_
session_df.to_csv('data/clustered_sessions.csv', index=False)
# Also save the scaler and cluster centroids for analysis in the app
joblib.dump(scaler, 'cluster_scaler.joblib')
joblib.dump(kmeans, 'cluster_model.joblib')


print("Clustering data and model saved successfully.")
print("\n--- All artifacts saved! Ready to build the Streamlit app. ---")