
from flask import Flask, request, jsonify
from predict import predict_charges
import math

# STEP 1: Initialize Flask application
app = Flask(__name__)

# Required model input features
FEATURE_COLUMNS = [
    'age', 'gender', 'bmi', 'children',
    'smoker', 'region', 'medical_history',
    'family_medical_history', 'exercise_frequency',
    'occupation', 'coverage_level'
]

# STEP 2: Health-check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'success',
        'message': 'Insurance Prediction API is running'
    }), 200


# STEP 3: Real-time prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():

    # Read customer details from JSON request
    customer_data = request.get_json(silent=True)

    # Validate request structure
    if not isinstance(customer_data, dict):
        return jsonify({
            'status': 'error',
            'message': 'A JSON object is required'
        }), 400

    # Check for missing input fields
    missing_fields = [
        col for col in FEATURE_COLUMNS
        if col not in customer_data
    ]

    if missing_fields:
        return jsonify({
            'status': 'error',
            'message': 'Required fields are missing',
            'missing_fields': missing_fields
        }), 400

    print("NEW VALIDATION CODE IS RUNNING", flush=True)
    print("Received customer data:", customer_data, flush=True)

    # Validate numerical features
    
    numerical_ranges = {
        'age': (18, 65),
        'bmi': (18, 50),
        'children': (0, 5)
    }

    # Define allowed values for categorical features
    CATEGORICAL_VALUES = {
        'gender': {'male', 'female', 'unknown'},
        'smoker': {'yes', 'no'},
        'region': {'northeast', 'northwest','southeast', 'southwest'},
        'medical_history': {'diabetes', 'heart disease','high blood pressure', 'unknown'},
        'family_medical_history': {'diabetes', 'heart disease','high blood pressure', 'unknown'},
        'exercise_frequency': {'never', 'rarely','occasionally', 'frequently'},
        'occupation': {'unemployed', 'student','blue collar', 'white collar', 'unknown'},
        'coverage_level': {'basic', 'standard', 'premium'}
    }
    
    # Features where unavailable information is supported
    UNKNOWN_ALLOWED = {'gender','medical_history','family_medical_history','occupation'}

    errors = {}
    
    for feature, (min_value, max_value) in numerical_ranges.items():
    
        value = customer_data[feature]
    
        # Check whether the value is numeric and finite
        # Explicit type check also rejects boolean values
        if type(value) not in (int, float) or not math.isfinite(value):
            errors[feature] = f'{feature} must be a valid number'
            continue
            # return jsonify({
            #     'status': 'error',
            #     'message': f'{feature} must be a valid number'
            # }), 400
    
        # Age and number of children must be whole numbers
        if feature in ['age', 'children'] and not float(value).is_integer():
            errors[feature] = f'{feature} must be an integer'
            continue
            # return jsonify({
            #     'status': 'error',
            #     'message': f'{feature} must be an integer'
            # }), 400
    
        # Validate allowed range
        if not min_value <= value <= max_value:
            errors[feature] = (
                f'{feature} must be between {min_value} and {max_value}'
                )
            # return jsonify({
            #     'status': 'error',
            #     'message': (
            #         f'{feature} must be between '
            #         f'{min_value} and {max_value}'
            #     )
            # }), 400

    for feature, allowed_values in CATEGORICAL_VALUES.items():
        
        value = customer_data[feature]

        # Normalize case and spaces
        normalized_value = value.strip().lower()

        # Convert JSON null to the supported Unknown category
        if value is None and feature in UNKNOWN_ALLOWED:
            customer_data[feature] = 'Unknown'
            continue
    
        # Validate categorical value
        if not isinstance(value, str) : # or value not in allowed_values
    
            errors[feature] = (
                f"{feature} must be one of: "
                f"{', '.join(sorted(allowed_values))}"
            )
            continue

        # Validate and convert to canonical form
        # if normalized_value in allowed_values:
        #     customer_data[feature] = allowed_map[normalized_value]
        # else:
        #     errors[feature] = (
        #         f"{feature} must be one of: "
        #         f"{', '.join(sorted(set(allowed_map.values())))}"
        #     )
        
        # Find matching category, ignoring case
        matched_value = next(
            (
                category for category in allowed_values
                if category.lower() == normalized_value
            ),
            None
        )
        
        if matched_value is not None:
            # Convert input to original training category
            customer_data[feature] = matched_value
        else:
            errors[feature] = (
                f"{feature} must be one of: "
                f"{', '.join(sorted(allowed_values))}"
            )

    # If any validation errors exist, return all of them together
    if errors:
        return jsonify({
            'status': 'error',
            'message': 'Invalid input values',
            'errors': errors
            }), 400

    # Generate predicted charges
    try:
        predicted_charges = predict_charges(customer_data)

        return jsonify({
            'status': 'success',
            'predicted_charges': predicted_charges
        }), 200

    except (ValueError, TypeError):
        return jsonify({
            'status': 'error',
            'message': 'Invalid customer input values'
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            'status': 'error',
            'message': 'Prediction failed'
        }), 500

    
        
# STEP 4: Run Flask development server
if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False
    )
