"""
Hospital Readmission Risk Assessment Tool

A Streamlit webapp that allows hospital staff to assess the readmission risk
of diabetic patients at discharge time.

Use Case:
- When a diabetic patient is being discharged, clinical staff enter patient data
- The app predicts the probability of readmission within 30 days
- Risk level and recommendations guide discharge planning decisions

This is a demonstration tool for the HSLU DSPRO1 project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from preprocessing import (
    load_model_artifacts,
    preprocess_input,
    get_risk_level,
    get_recommendations,
    ICD9_CATEGORIES,
    AGE_BRACKETS,
    RACE_OPTIONS,
    ADMISSION_TYPES,
    DISCHARGE_DISPOSITIONS,
    ADMISSION_SOURCES,
    MEDICAL_SPECIALTIES,
    MEDICATION_OPTIONS,
    A1C_OPTIONS,
    GLUCOSE_OPTIONS
)

# Page configuration
st.set_page_config(
    page_title="Readmission Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .risk-high {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffa500;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model artifacts with caching for performance."""
    try:
        return load_model_artifacts()
    except FileNotFoundError:
        return None, None, None


def main():
    # Header
    st.title("🏥 Hospital Readmission Risk Assessment")
    st.markdown("""
    **Diabetic Patient Discharge Planning Tool**

    This tool helps clinical staff assess the risk of hospital readmission for diabetic patients
    at discharge time. Enter patient information below to receive a risk assessment and
    recommendations for discharge planning.
    """)

    st.divider()

    # Load model
    model, scaler, feature_columns = load_model()

    if model is None:
        st.error("""
        **Model not found!**

        Please run the export script first to generate the model files:
        ```
        cd hospital-readmission-predictor
        python webapp/export_model.py
        ```
        """)
        st.stop()

    # Create input form
    st.header("Patient Information")

    # Use columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.selectbox("Age Group", AGE_BRACKETS, index=6)  # Default [60-70)
        race = st.selectbox("Race", RACE_OPTIONS)

    with col2:
        st.subheader("Admission Details")
        admission_type = st.selectbox("Admission Type", ADMISSION_TYPES)
        admission_source = st.selectbox("Admission Source", ADMISSION_SOURCES)
        discharge_disposition = st.selectbox(
            "Discharge Disposition",
            DISCHARGE_DISPOSITIONS
        )

    with col3:
        st.subheader("Medical Specialty")
        medical_specialty = st.selectbox("Primary Specialty", MEDICAL_SPECIALTIES)

    st.divider()

    # Diagnosis information
    st.header("Diagnosis Information")
    diag_col1, diag_col2, diag_col3 = st.columns(3)

    with diag_col1:
        diag_1 = st.selectbox(
            "Primary Diagnosis Category",
            ICD9_CATEGORIES,
            index=6,  # Default: Circulatory
            help="Main reason for hospitalization"
        )

    with diag_col2:
        diag_2 = st.selectbox(
            "Secondary Diagnosis Category",
            ICD9_CATEGORIES,
            index=2,  # Default: Endocrine
            help="Second diagnosis"
        )

    with diag_col3:
        diag_3 = st.selectbox(
            "Tertiary Diagnosis Category",
            ICD9_CATEGORIES,
            index=6,  # Default: Circulatory
            help="Third diagnosis"
        )

    st.divider()

    # Clinical measurements
    st.header("Clinical Information")
    clin_col1, clin_col2, clin_col3, clin_col4 = st.columns(4)

    with clin_col1:
        time_in_hospital = st.number_input(
            "Days in Hospital",
            min_value=1, max_value=14, value=4,
            help="Length of current hospital stay"
        )
        num_lab_procedures = st.number_input(
            "Lab Procedures",
            min_value=0, max_value=132, value=40,
            help="Number of lab tests performed"
        )

    with clin_col2:
        num_procedures = st.number_input(
            "Other Procedures",
            min_value=0, max_value=6, value=1,
            help="Number of non-lab procedures"
        )
        num_medications = st.number_input(
            "Total Medications",
            min_value=0, max_value=81, value=15,
            help="Total number of medications"
        )

    with clin_col3:
        number_diagnoses = st.number_input(
            "Number of Diagnoses",
            min_value=1, max_value=16, value=7,
            help="Total diagnoses on record"
        )
        a1c_result = st.selectbox(
            "HbA1c Result",
            list(A1C_OPTIONS.keys()),
            help="Hemoglobin A1c test result"
        )

    with clin_col4:
        glucose_serum = st.selectbox(
            "Glucose Serum",
            list(GLUCOSE_OPTIONS.keys()),
            help="Serum glucose test result"
        )
        change = st.checkbox("Diabetes Medication Changed", value=False)
        diabetesMed = st.checkbox("On Diabetes Medication", value=True)

    st.divider()

    # Prior healthcare utilization
    st.header("Prior Healthcare Utilization (Past Year)")
    util_col1, util_col2, util_col3 = st.columns(3)

    with util_col1:
        number_inpatient = st.number_input(
            "Prior Inpatient Visits",
            min_value=0, max_value=21, value=0,
            help="Hospital admissions in past year"
        )

    with util_col2:
        number_emergency = st.number_input(
            "Prior Emergency Visits",
            min_value=0, max_value=76, value=0,
            help="ER visits in past year"
        )

    with util_col3:
        number_outpatient = st.number_input(
            "Prior Outpatient Visits",
            min_value=0, max_value=42, value=0,
            help="Outpatient visits in past year"
        )

    st.divider()

    # Medication details (collapsible)
    with st.expander("Diabetes Medications Detail (Optional)", expanded=False):
        st.markdown("Specify individual diabetes medication status if known:")

        med_col1, med_col2, med_col3, med_col4 = st.columns(4)
        med_options = list(MEDICATION_OPTIONS.keys())

        with med_col1:
            metformin = st.selectbox("Metformin", med_options, key="metformin")
            glimepiride = st.selectbox("Glimepiride", med_options, key="glimepiride")
            pioglitazone = st.selectbox("Pioglitazone", med_options, key="pioglitazone")

        with med_col2:
            repaglinide = st.selectbox("Repaglinide", med_options, key="repaglinide")
            glipizide = st.selectbox("Glipizide", med_options, key="glipizide")
            rosiglitazone = st.selectbox("Rosiglitazone", med_options, key="rosiglitazone")

        with med_col3:
            nateglinide = st.selectbox("Nateglinide", med_options, key="nateglinide")
            glyburide = st.selectbox("Glyburide", med_options, key="glyburide")
            acarbose = st.selectbox("Acarbose", med_options, key="acarbose")

        with med_col4:
            chlorpropamide = st.selectbox("Chlorpropamide", med_options, key="chlorpropamide")
            insulin = st.selectbox("Insulin", med_options, index=2, key="insulin")  # Default: Steady

    st.divider()

    # Predict button
    if st.button("🔍 Assess Readmission Risk", type="primary", use_container_width=True):

        # Gather all patient data
        patient_data = {
            'gender': gender,
            'age': age,
            'race': race,
            'admission_type': admission_type,
            'admission_source': admission_source,
            'discharge_disposition': discharge_disposition,
            'medical_specialty': medical_specialty,
            'diag_1_category': diag_1,
            'diag_2_category': diag_2,
            'diag_3_category': diag_3,
            'time_in_hospital': time_in_hospital,
            'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'number_diagnoses': number_diagnoses,
            'number_inpatient': number_inpatient,
            'number_emergency': number_emergency,
            'number_outpatient': number_outpatient,
            'A1Cresult': a1c_result,
            'max_glu_serum': glucose_serum,
            'change': change,
            'diabetesMed': diabetesMed,
            # Medications
            'metformin': metformin,
            'repaglinide': repaglinide,
            'nateglinide': nateglinide,
            'chlorpropamide': chlorpropamide,
            'glimepiride': glimepiride,
            'glipizide': glipizide,
            'glyburide': glyburide,
            'pioglitazone': pioglitazone,
            'rosiglitazone': rosiglitazone,
            'acarbose': acarbose,
            'insulin': insulin
        }

        # Preprocess and predict
        with st.spinner("Analyzing patient data..."):
            try:
                X = preprocess_input(patient_data, scaler, feature_columns)
                probability = model.predict_proba(X)[0, 1]
                risk_level, risk_color, risk_desc = get_risk_level(probability)
                recommendations = get_recommendations(probability, patient_data)

                # Display results
                st.header("Risk Assessment Results")

                # Risk score display
                result_col1, result_col2 = st.columns([1, 2])

                with result_col1:
                    st.metric(
                        label="Readmission Probability",
                        value=f"{probability*100:.1f}%"
                    )

                    # Progress bar for visual
                    st.progress(probability)

                with result_col2:
                    risk_class = f"risk-{risk_color.lower().replace('orange', 'medium').replace('green', 'low').replace('red', 'high')}"
                    st.markdown(f"""
                    <div class="{risk_class}">
                        <h3 style="margin-top:0;">{risk_level}</h3>
                        <p>{risk_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()

                # Recommendations
                st.header("Discharge Planning Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"**{i}.** {rec}")

                # Summary statistics
                st.divider()
                st.subheader("Patient Summary")

                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

                with sum_col1:
                    st.metric("Hospital Stay", f"{time_in_hospital} days")

                with sum_col2:
                    total_visits = number_inpatient + number_emergency + number_outpatient
                    st.metric("Prior Visits (Year)", total_visits)

                with sum_col3:
                    st.metric("Total Medications", num_medications)

                with sum_col4:
                    st.metric("Diagnoses", number_diagnoses)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please ensure all fields are filled correctly.")

    # Footer
    st.divider()
    st.markdown("""
    ---
    **Disclaimer**: This tool is for demonstration purposes only. Clinical decisions should always
    involve professional medical judgment. This model was trained on historical data (1999-2008)
    and may not reflect current clinical practices.

    *HSLU DSPRO1 - Hospital Readmission Prediction Project*
    """)


if __name__ == "__main__":
    main()
