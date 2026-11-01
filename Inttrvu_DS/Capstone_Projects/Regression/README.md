
# Insurance Charges Prediction using Machine Learning and Flask

## Project Overview

This project develops an end-to-end Machine Learning application
to predict insurance charges based on customer demographic,
medical, lifestyle, and coverage information.

The solution uses Ridge Regression and provides real-time
predictions through a Flask REST API.

## Dataset

- Total records: 1,000,000
- Input features: 11
- Target variable: charges
- Data source: SQLite database

### Dataset Allocation

| Dataset | Records | Purpose |
|---|---:|---|
| Training | 700,000 | Model development |
| Evaluation | 200,000 | Final model testing |
| Live | 100,000 | Production prediction simulation |

## Feature Engineering

The preprocessing pipeline includes:

- Median imputation for numerical features
- Unknown-category imputation for missing categorical values
- StandardScaler for numerical standardization
- One-Hot Encoding for categorical variables
- Consistent preprocessing during training and prediction

## Models Evaluated

- Linear Regression
- Ridge Regression
- Decision Tree Regression
- Random Forest Regression
- Histogram-based Gradient Boosting Regression

### Selected Model: Ridge Regression

Hyperparameter: alpha = 1.0

### Final Evaluation Results

| Metric | Result |
|---|---:|
| MAE | 275.6770 |
| RMSE | 340.4687 |
| R² | 0.9941 |

The selected model achieved an R² score of 99.41%
on the reserved evaluation dataset.

## Project Structure

    Regression/
    ├── train.py
    ├── predict.py
    ├── app.py
    ├── requirements.txt
    ├── .gitignore
    ├── README.md
    ├── regression.db
    └── models/
        └── insurance_pipeline.joblib

Note: regression.db is excluded from Git.
Obtain the original database separately to retrain the model.

## Installation

Install the dependencies:

    pip install -r requirements.txt

The saved model must be available at:

    models/insurance_pipeline.joblib

## Model Training

To retrain from the SQLite database:

    python train.py

## Run the Flask API

Start the Flask development server:

    python app.py

The API will be accessible at:

    http://127.0.0.1:5000

## API Endpoints

### GET /health

Checks whether the API is running.

### POST /predict

Accepts customer details and returns predicted insurance charges.

Example request:

    {
      "age": 35,
      "gender": "female",
      "bmi": 27.5,
      "children": 2,
      "smoker": "no",
      "region": "southeast",
      "medical_history": "Diabetes",
      "family_medical_history": "Heart disease",
      "exercise_frequency": "Frequently",
      "occupation": "White collar",
      "coverage_level": "Premium"
    }

Example response:

    {
      "status": "success",
      "predicted_charges": 20977.35
    }

## API Validation

The Flask API supports:

- Required-field validation
- Numerical input range validation
- Multiple validation errors in one response
- Case-insensitive categorical input
- Leading and trailing whitespace handling
- HTTP 400 responses for invalid requests

## Production Prediction Simulation

The trained model successfully processed 100,000
reserved live records.

- Missing predictions: 0
- Negative predictions: 0
- Infinite predictions: 0

## Important Notes

The target variable is named charges. Predictions should
not be treated as binding insurance quotations.

The Flask development server is intended for local testing.
Production deployment requires a production-grade server,
security controls, and appropriate model governance.
