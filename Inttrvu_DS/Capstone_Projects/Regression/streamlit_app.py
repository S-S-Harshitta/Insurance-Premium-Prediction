
import streamlit as st
from predict import predict_charges

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Insurance Premium Prediction")
st.write(
    "Predict insurance charges using a trained "
    "Ridge Regression machine learning model."
)

# Collect customer details
with st.form("insurance_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=18, max_value=65, value=35
        )
        gender = st.selectbox(
            "Gender", ["male", "female", "Unknown"]
        )
        bmi = st.number_input(
            "BMI", min_value=18.0, max_value=50.0,
            value=27.5, step=0.1
        )
        children = st.number_input(
            "Number of Children", min_value=0,
            max_value=5, value=2
        )
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"]
        )

    with col2:
        medical_history = st.selectbox(
            "Medical History",
            ["Unknown", "Diabetes", "Heart disease",
             "High blood pressure"]
        )
        family_medical_history = st.selectbox(
            "Family Medical History",
            ["Unknown", "Diabetes", "Heart disease",
             "High blood pressure"]
        )
        exercise_frequency = st.selectbox(
            "Exercise Frequency",
            ["Never", "Rarely", "Occasionally", "Frequently"]
        )
        occupation = st.selectbox(
            "Occupation",
            ["Unemployed", "Student", "Blue collar",
             "White collar", "Unknown"]
        )
        coverage_level = st.selectbox(
            "Coverage Level",
            ["Basic", "Standard", "Premium"]
        )

    submitted = st.form_submit_button(
        "Predict Insurance Charges",
        use_container_width=True
    )

if submitted:

    customer_data = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "medical_history": medical_history,
        "family_medical_history": family_medical_history,
        "exercise_frequency": exercise_frequency,
        "occupation": occupation,
        "coverage_level": coverage_level
    }

    try:
        prediction = predict_charges(customer_data)

        st.success("Prediction completed successfully!")
        st.metric(
            label="Estimated Insurance Charges",
            value=f"{prediction:,.2f}"
        )

        st.caption(
            "Model estimate only. Not an actual insurance quote."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
