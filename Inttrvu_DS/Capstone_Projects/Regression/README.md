
# Insurance Charges Prediction using Machine Learning and Flask

## Project Overview

This project develops an end-to-end Machine Learning application
to predict insurance charges based on customer demographic,
medical, lifestyle, and coverage information.

The solution uses Ridge Regression and provides real-time
prediction API and a web interface for real-time inference.

The app is deployed here:

**Live Demo** : https://insurance-premium-prediction-ebsmfumghgivxfyz6j9udu.streamlit.app/?utm_source=chatgpt.com

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

## Methodology

1. **Data Cleaning** :
    - Processed 1 million insurance records
    - Validated numerical ranges
    - Checked for duplicates
    - Identified missing values across demographic, medical, and lifestyle features

2. **Exploratory Data Analysis (EDA)** : 
    - Analyzed insurance charge distributions
    - Identified outliers
    - Investigated relationships between charges and key factors such as smoking status, age, BMI, medical history, and coverage level

3. **Feature Engineering** : 
   - Median imputation for numerical features
   - Unknown-category imputation for missing categorical values
   - StandardScaler for numerical standardization
   - One-Hot Encoding for categorical variables
   - Consistent preprocessing during training and prediction

4. **Model Development & Evaluation** :  
    - Models Evaluated
      - Linear Regression
      - Ridge Regression
      - Decision Tree Regression
      - Random Forest Regression
      - Histogram-based Gradient Boosting Regression
    - Compared performance using MAE, RMSE, and R²
      
5. **Optimization & Deployment** – 
    - Selected Model: Ridge Regression
      - Hyperparameter: alpha = 1.0
    - Final Evaluation Results
      
        | Metric | Result |
        |---|---:|
        | MAE | 275.6770 |
        | RMSE | 340.4687 |
        | R² | 0.9941 |

## Key Findings

1. **Smoking Impact:** Smokers incur approximately 35% higher average insurance charges than non-smokers.

2. **Outlier Concentration:** All 2,190 upper-charge outliers are associated with smokers.

3. **Balanced Dataset:** Gender and regional distributions are nearly uniform, providing balanced representation across these categories.

4. **Data Quality:** Missing values of 2.4%–4% in selected features were addressed through numerical and categorical imputation.

## Project Structure

    Regression/
    ├── train.py
    ├── predict.py
    ├── streamlit_app.py
    ├── requirements.txt
    ├── .gitignore
    ├── README.md
    ├── regression.db
    └── models/
        └── insurance_pipeline.joblib

## Tech Stack
- Jupyter Notebook, Python, pandas, NumPy, scikit-learn, joblib (ML Model Development)
- Streamlit (App Deployment)
- SQLite (Data Source)
