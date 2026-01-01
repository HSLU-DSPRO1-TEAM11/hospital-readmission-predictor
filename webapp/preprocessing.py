"""
Preprocessing Module for Webapp

Transforms raw patient input data into model-ready features.
This replicates the feature engineering pipeline from training.

Reasoning:
- The model expects exactly 180 features in a specific order
- User inputs are in human-readable format (dropdowns, numbers)
- This module bridges the gap between user input and model input
- We use the same transformations as src/feature_engineering.py
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import re

# ICD-9 diagnosis categories (from feature_engineering.py)
ICD9_CATEGORIES = [
    'Infectious and parasitic diseases',
    'Neoplasms',
    'Endocrine, nutritional and metabolic diseases, and immunity disorders',
    'Diseases of the blood and blood-forming organs',
    'Mental disorders',
    'Diseases of the nervous system and sense organs',
    'Diseases of the circulatory system',
    'Diseases of the respiratory system',
    'Diseases of the digestive system',
    'Diseases of the genitourinary system',
    'Complications of pregnancy, childbirth, and the puerperium',
    'Diseases of the skin and subcutaneous tissue',
    'Diseases of the musculoskeletal system and connective tissue',
    'Congenital anomalies',
    'Certain conditions originating in the perinatal period',
    'Symptoms, signs, and ill-defined conditions',
    'Injury and poisoning',
    'External causes of injury',
    'Supplemental classification',
    'Unknown'
]

# Age brackets
AGE_BRACKETS = [
    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
]

# Race options
RACE_OPTIONS = ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other']

# Admission types
ADMISSION_TYPES = [
    'Emergency', 'Urgent', 'Elective', 'Newborn', 'Trauma Center', 'Not Available'
]

# Discharge dispositions
DISCHARGE_DISPOSITIONS = [
    'Discharged to home',
    'Discharged/transferred to SNF',
    'Discharged/transferred to another short term hospital',
    'Discharged/transferred to home with home health service',
    'Left AMA',
    'Discharged/transferred to another rehab fac including rehab units of a hospital',
    'Discharged/transferred to ICF',
    'Discharged/transferred to a long term care hospital.',
    'Unknown/Invalid',
    'Expired',
    'Hospice / home',
    'Hospice / medical facility',
    'Discharged/transferred/referred another institution for outpatient services',
    'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
    'Not Mapped',
    'Discharged/transferred within this institution to Medicare approved swing bed',
    'Discharged/transferred/referred to this institution for outpatient services',
    'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
    'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
    'Discharged/transferred to a federal health care facility.',
    'Still patient or expected to return for outpatient services'
]

# Admission sources
ADMISSION_SOURCES = [
    'Physician Referral',
    'Emergency Room',
    'Transfer from a hospital',
    'Transfer from a Skilled Nursing Facility (SNF)',
    'Transfer from another health care facility',
    'Clinic Referral',
    'HMO Referral',
    'Court/Law Enforcement',
    'Not Available'
]

# Medical specialties (top ones + other)
MEDICAL_SPECIALTIES = [
    'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
    'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
    'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology',
    'Psychiatry', 'Urology', 'Gastroenterology', 'Surgery-Cardiovascular/Thoracic',
    'Hematology/Oncology', 'Neurology', 'Oncology', 'Other'
]

# Medication options
MEDICATION_OPTIONS = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}

# A1C result options
A1C_OPTIONS = {'Not tested': 0, 'Normal': 1, '>7': 2, '>8': 3}

# Glucose serum options
GLUCOSE_OPTIONS = {'Not tested': 0, 'Normal': 1, '>200': 2, '>300': 3}


def load_model_artifacts(model_dir='model'):
    """Load the trained model, scaler, and feature columns."""
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, model_dir)

    with open(os.path.join(model_path, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(model_path, 'feature_columns.json'), 'r') as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


def preprocess_input(patient_data: dict, scaler, feature_columns: list) -> pd.DataFrame:
    """
    Transform patient input data into model-ready features.

    Args:
        patient_data: Dictionary with raw patient information
        scaler: Fitted StandardScaler for numeric columns
        feature_columns: List of expected feature column names

    Returns:
        DataFrame with exactly the features the model expects
    """
    # Start with a DataFrame of zeros for all features
    df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # --- Numeric features (scaled) ---
    numeric_cols = ['num_lab_procedures', 'num_procedures', 'total_visits']

    # Calculate total_visits from component values
    total_visits = (
        patient_data.get('number_inpatient', 0) +
        patient_data.get('number_emergency', 0) +
        patient_data.get('number_outpatient', 0)
    )

    # Create array for scaling
    numeric_values = np.array([[
        patient_data.get('num_lab_procedures', 0),
        patient_data.get('num_procedures', 0),
        total_visits
    ]])

    # Apply scaler
    scaled_values = scaler.transform(numeric_values)

    for i, col in enumerate(numeric_cols):
        if col in df.columns:
            df[col] = scaled_values[0, i]

    # --- Binary/Ordinal features ---
    # Gender (Male=0, Female=1)
    if 'gender' in df.columns:
        df['gender'] = 1 if patient_data.get('gender') == 'Female' else 0

    # Change (dosage changed)
    if 'change' in df.columns:
        df['change'] = 1 if patient_data.get('change', False) else 0

    # DiabetesMed
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = 1 if patient_data.get('diabetesMed', False) else 0

    # A1C result
    if 'A1Cresult' in df.columns:
        df['A1Cresult'] = A1C_OPTIONS.get(patient_data.get('A1Cresult', 'Not tested'), 0)

    # Glucose serum
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum'] = GLUCOSE_OPTIONS.get(patient_data.get('max_glu_serum', 'Not tested'), 0)

    # --- Medication columns ---
    medications = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'insulin'
    ]

    for med in medications:
        if med in df.columns:
            med_value = patient_data.get(med, 'No')
            df[med] = MEDICATION_OPTIONS.get(med_value, 0)

    # --- Derived features ---
    # Polypharmacy level (based on num_medications)
    num_meds = patient_data.get('num_medications', 0)
    if 'polypharmacy_level' in df.columns:
        if num_meds <= 5:
            df['polypharmacy_level'] = 0
        elif num_meds <= 10:
            df['polypharmacy_level'] = 1
        elif num_meds <= 20:
            df['polypharmacy_level'] = 2
        else:
            df['polypharmacy_level'] = 3

    # Comorbidity score (based on number_diagnoses)
    num_diag = patient_data.get('number_diagnoses', 0)
    if 'comorbidity_score' in df.columns:
        if num_diag <= 3:
            df['comorbidity_score'] = 0
        elif num_diag <= 6:
            df['comorbidity_score'] = 1
        elif num_diag <= 9:
            df['comorbidity_score'] = 2
        else:
            df['comorbidity_score'] = 3

    # Hospital visit flags
    if 'had_inpatient' in df.columns:
        df['had_inpatient'] = 1 if patient_data.get('number_inpatient', 0) > 0 else 0
    if 'had_emergency' in df.columns:
        df['had_emergency'] = 1 if patient_data.get('number_emergency', 0) > 0 else 0
    if 'had_outpatient' in df.columns:
        df['had_outpatient'] = 1 if patient_data.get('number_outpatient', 0) > 0 else 0

    # Frequent visitor
    if 'frequent_visitor' in df.columns:
        df['frequent_visitor'] = 1 if total_visits >= 5 else 0

    # Stay length category
    time_in_hospital = patient_data.get('time_in_hospital', 1)
    if 'stay_length_cat' in df.columns:
        if time_in_hospital <= 3:
            df['stay_length_cat'] = 0
        elif time_in_hospital <= 6:
            df['stay_length_cat'] = 1
        elif time_in_hospital <= 10:
            df['stay_length_cat'] = 2
        else:
            df['stay_length_cat'] = 3

    # --- One-hot encoded categorical features ---
    # Helper function to set one-hot column
    def set_onehot(prefix, value):
        """Set the appropriate one-hot column for a categorical value."""
        # Clean the value to match column naming
        clean_value = re.sub(r'[^A-Za-z0-9_]+', '_', str(value))
        col_name = f"{prefix}_{clean_value}"

        # Check if this column exists
        if col_name in df.columns:
            df[col_name] = 1
        else:
            # Try variations (different cleaning patterns)
            for col in df.columns:
                if col.startswith(prefix + '_') and clean_value.lower() in col.lower():
                    df[col] = 1
                    break

    # Race
    set_onehot('race', patient_data.get('race', 'Caucasian'))

    # Age
    set_onehot('age', patient_data.get('age', '[50-60)'))

    # Admission type
    set_onehot('admission_type', patient_data.get('admission_type', 'Emergency'))

    # Discharge disposition
    set_onehot('discharge_disposition', patient_data.get('discharge_disposition', 'Discharged to home'))

    # Admission source
    set_onehot('admission_source', patient_data.get('admission_source', 'Emergency Room'))

    # Diagnosis categories
    set_onehot('diag_1_category', patient_data.get('diag_1_category', 'Diseases of the circulatory system'))
    set_onehot('diag_2_category', patient_data.get('diag_2_category', 'Diseases of the circulatory system'))
    set_onehot('diag_3_category', patient_data.get('diag_3_category', 'Diseases of the circulatory system'))

    # Medical specialty
    set_onehot('medical_specialty', patient_data.get('medical_specialty', 'InternalMedicine'))

    return df


def get_risk_level(probability: float) -> tuple:
    """
    Categorize readmission probability into risk levels.

    Returns:
        Tuple of (level_name, color, description)
    """
    if probability < 0.3:
        return ('Low Risk', 'green', 'Patient has lower likelihood of readmission.')
    elif probability < 0.6:
        return ('Medium Risk', 'orange', 'Patient has moderate likelihood of readmission. Consider enhanced follow-up.')
    else:
        return ('High Risk', 'red', 'Patient has elevated likelihood of readmission. Recommend comprehensive discharge planning.')


def get_recommendations(probability: float, patient_data: dict) -> list:
    """
    Generate recommendations based on risk level and patient characteristics.

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Base recommendations on risk level
    if probability >= 0.6:
        recommendations.append("Schedule follow-up appointment within 7 days of discharge")
        recommendations.append("Consider care coordinator involvement")
        recommendations.append("Ensure medication reconciliation is completed")
        recommendations.append("Provide detailed discharge instructions with teach-back")
    elif probability >= 0.3:
        recommendations.append("Schedule follow-up appointment within 14 days")
        recommendations.append("Review and simplify medication regimen if possible")
        recommendations.append("Ensure patient has pharmacy access for prescriptions")
    else:
        recommendations.append("Standard discharge protocol appropriate")
        recommendations.append("Schedule routine follow-up within 30 days")

    # Specific recommendations based on patient factors
    if patient_data.get('number_inpatient', 0) > 2:
        recommendations.append("Patient has history of multiple hospitalizations - consider case management referral")

    if patient_data.get('number_diagnoses', 0) >= 7:
        recommendations.append("Multiple comorbidities present - coordinate with specialists")

    if patient_data.get('num_medications', 0) > 15:
        recommendations.append("High medication burden - pharmacist consultation recommended")

    total_visits = (
        patient_data.get('number_inpatient', 0) +
        patient_data.get('number_emergency', 0) +
        patient_data.get('number_outpatient', 0)
    )
    if total_visits >= 5:
        recommendations.append("Frequent healthcare utilizer - assess for unmet care needs")

    if patient_data.get('time_in_hospital', 0) > 7:
        recommendations.append("Extended hospital stay - ensure adequate transition support")

    return recommendations
