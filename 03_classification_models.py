import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow # <-- NEW: Import mlflow
import mlflow.sklearn # <-- NEW: Import the scikit-learn flavor

# --- Load the Processed Data ---
X_train = pd.read_csv('data/X_train.csv')
X_val = pd.read_csv('data/X_val.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_val = pd.read_csv('data/y_val.csv').values.ravel()

print("--- Data loaded successfully ---")

# --- NEW: Start an MLflow Run ---
with mlflow.start_run():
    # --- Build the Logistic Regression Pipeline ---
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', LogisticRegression(random_state=42, max_iter=1000)) # Added max_iter for convergence
    ])

    print("--- Starting model training ---")
    pipeline.fit(X_train, y_train)
    print("--- Model training complete ---")

    # --- Evaluate the Model ---
    print("\n--- Evaluating model on the validation set ---")
    y_pred = pipeline.predict(X_val)

    # --- NEW: Log Metrics to MLflow ---
    # We calculate metrics and then log them
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    mlflow.log_metric("validation_precision", precision)
    mlflow.log_metric("validation_recall", recall)
    mlflow.log_metric("validation_f1_score", f1)
    
    # --- NEW: Log the Model Itself ---
    mlflow.sklearn.log_model(pipeline, "logistic_regression_model")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print("\n--- Experiment run logged to MLflow! ---")

    # --- Everything from the Logistic Regression model goes above this line ---
from sklearn.ensemble import RandomForestClassifier

# --- Build the Random Forest Pipeline ---
print("\n\n--- Starting Random Forest Model ---")

# NEW: Start a new MLflow run with a descriptive name
with mlflow.start_run(run_name="RandomForest_Model"):

    # We log the model type as a parameter for easy filtering in MLflow
    mlflow.log_param("model_type", "RandomForest")

    # Define the new pipeline with RandomForestClassifier
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(random_state=42, n_jobs=-1)) # n_jobs=-1 uses all available CPU cores
    ])

    print("--- Starting model training ---")
    rf_pipeline.fit(X_train, y_train)
    print("--- Model training complete ---")

    # --- Evaluate the Model ---
    print("\n--- Evaluating Random Forest model on the validation set ---")
    y_pred_rf = rf_pipeline.predict(X_val)

    # Log metrics to MLflow
    precision_rf = precision_score(y_val, y_pred_rf)
    recall_rf = recall_score(y_val, y_pred_rf)
    f1_rf = f1_score(y_val, y_pred_rf)

    mlflow.log_metric("validation_precision", precision_rf)
    mlflow.log_metric("validation_recall", recall_rf)
    mlflow.log_metric("validation_f1_score", f1_rf)

    # Log the Random Forest model
    mlflow.sklearn.log_model(rf_pipeline, "random_forest_model")

    print("\nRandom Forest Classification Report:")
    print(classification_report(y_val, y_pred_rf))
    print("\n--- Random Forest experiment run logged to MLflow! ---")

from xgboost import XGBClassifier

# --- Build the XGBoost Pipeline ---
print("\n\n--- Starting XGBoost Model ---")

# Start a new MLflow run for XGBoost
with mlflow.start_run(run_name="XGBoost_Model"):

    mlflow.log_param("model_type", "XGBoost")

    # Define the new pipeline with XGBClassifier
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')) # Common settings for XGBoost
    ])

    print("--- Starting model training ---")
    xgb_pipeline.fit(X_train, y_train)
    print("--- Model training complete ---")

    # --- Evaluate the Model ---
    print("\n--- Evaluating XGBoost model on the validation set ---")
    y_pred_xgb = xgb_pipeline.predict(X_val)

    # Log metrics to MLflow
    precision_xgb = precision_score(y_val, y_pred_xgb)
    recall_xgb = recall_score(y_val, y_pred_xgb)
    f1_xgb = f1_score(y_val, y_pred_xgb)

    mlflow.log_metric("validation_precision", precision_xgb)
    mlflow.log_metric("validation_recall", recall_xgb)
    mlflow.log_metric("validation_f1_score", f1_xgb)

    # Log the XGBoost model
    mlflow.sklearn.log_model(xgb_pipeline, "xgboost_model")

    print("\nXGBoost Classification Report:")
    print(classification_report(y_val, y_pred_xgb))
    print("\n--- XGBoost experiment run logged to MLflow! ---")