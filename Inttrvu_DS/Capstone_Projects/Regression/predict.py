
import joblib
import pandas as pd
from pathlib import Path

# Load saved model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "insurance_pipeline.joblib"
model = joblib.load(MODEL_PATH)

# Function for predicting insurance charges
def predict_charges(customer_details):
    customer_data = pd.DataFrame([customer_details])

    prediction = model.predict(customer_data)

    return round(float(prediction[0]), 2)


# Test the prediction function
if __name__ == "__main__":

    customer = {
        'age': 35,
        'gender': 'female',
        'bmi': 27.5,
        'children': 2,
        'smoker': 'no',
        'region': 'southeast',
        'medical_history': 'Diabetes',
        'family_medical_history': 'Heart disease',
        'exercise_frequency': 'Frequently',
        'occupation': 'White collar',
        'coverage_level': 'Premium'
    }

    result = predict_charges(customer)
    print(f"Predicted Insurance Charges: {result:.2f}")
