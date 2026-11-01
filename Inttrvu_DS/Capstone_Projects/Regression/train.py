
from pathlib import Path
import joblib
import sqlite3
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Define file paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "regression.db"
MODEL_PATH = BASE_DIR / "models" / "insurance_pipeline.joblib"

# STEP 2: Connect to SQLite and read original data  
with sqlite3.connect(DB_PATH) as regression_db:
    df = pd.read_sql_query(
        "SELECT * FROM Insurance_Prediction",
        regression_db
    )

print("Dataset shape:", df.shape)

# STEP 3: Define features
numerical_cols = ['age', 'bmi', 'children']

categorical_cols = [
    'gender', 'smoker', 'region',
    'medical_history', 'family_medical_history',
    'exercise_frequency', 'occupation', 'coverage_level'
]

feature_cols = numerical_cols + categorical_cols

# STEP 4: Split according to assignment requirements
if len(df) != 1_000_000:
    raise ValueError("Expected exactly 1,000,000 records")

train_df = df.iloc[:700_000].copy()
eval_df = df.iloc[700_000:900_000].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df['charges']

X_eval = eval_df[feature_cols].copy()
y_eval = eval_df['charges']

# STEP 5: Normalize categorical missing values
# Ensure both None and NaN are represented as Python None
for col in categorical_cols:
    X_train[col] = X_train[col].astype(object).where(
        X_train[col].notna(), None
    )
    X_eval[col] = X_eval[col].astype(object).where(
        X_eval[col].notna(), None
    )

# STEP 6: Numerical preprocessing
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# STEP 7: Categorical preprocessing
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(
        missing_values=None,
        strategy='constant',
        fill_value='Unknown'
    )),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# STEP 8: Combine preprocessing
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# STEP 9: Create model pipeline
final_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', Ridge(alpha=1.0))
])

# STEP 10: Train model
final_pipeline.fit(X_train, y_train)
print("Model training completed!")

# STEP 11: Evaluate model
predictions = final_pipeline.predict(X_eval)

mae = mean_absolute_error(y_eval, predictions)
rmse = np.sqrt(mean_squared_error(y_eval, predictions))
r2 = r2_score(y_eval, predictions)

print("\nFinal Model Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# STEP 12: Save fitted pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final_pipeline, MODEL_PATH)

print("\nModel saved successfully!")
print("Model path:", MODEL_PATH)
