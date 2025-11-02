import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


from src.data_cleaning import clean_data
df = clean_data()


def categorize_icd9(code):
    """Map ICD-9 diagnosis codes to major medical categories."""
    if pd.isna(code):
        return 'Unknown'
    code_str = str(code)

    # Handle E and V codes first
    if code_str.startswith('E'):
        return 'External causes of injury'
    elif code_str.startswith('V'):
        return 'Supplemental classification'

    # Try numeric conversion for standard ICD-9 codes
    try:
        code_val = float(code_str.split('.')[0])
    except ValueError:
        return 'Unknown'

    # Map ranges to categories
    if 1 <= code_val <= 139:
        return 'Infectious and parasitic diseases'
    elif 140 <= code_val <= 239:
        return 'Neoplasms'
    elif 240 <= code_val <= 279:
        return 'Endocrine, nutritional and metabolic diseases, and immunity disorders'
    elif 280 <= code_val <= 289:
        return 'Diseases of the blood and blood-forming organs'
    elif 290 <= code_val <= 319:
        return 'Mental disorders'
    elif 320 <= code_val <= 389:
        return 'Diseases of the nervous system and sense organs'
    elif 390 <= code_val <= 459:
        return 'Diseases of the circulatory system'
    elif 460 <= code_val <= 519:
        return 'Diseases of the respiratory system'
    elif 520 <= code_val <= 579:
        return 'Diseases of the digestive system'
    elif 580 <= code_val <= 629:
        return 'Diseases of the genitourinary system'
    elif 630 <= code_val <= 679:
        return 'Complications of pregnancy, childbirth, and the puerperium'
    elif 680 <= code_val <= 709:
        return 'Diseases of the skin and subcutaneous tissue'
    elif 710 <= code_val <= 739:
        return 'Diseases of the musculoskeletal system and connective tissue'
    elif 740 <= code_val <= 759:
        return 'Congenital anomalies'
    elif 760 <= code_val <= 779:
        return 'Certain conditions originating in the perinatal period'
    elif 780 <= code_val <= 799:
        return 'Symptoms, signs, and ill-defined conditions'
    elif 800 <= code_val <= 999:
        return 'Injury and poisoning'
    else:
        return 'Unknown'

# Apply to all 3 diagnosis columns
for col in ['diag_1', 'diag_2', 'diag_3']:
    df[f'{col}_category'] = df[col].apply(categorize_icd9)

df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['change'] = df['change'].astype(int)
df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)
df['readmitted'] = df['readmitted'].astype(int)

map_a1c = {
    0: 0,
    'None': 0,
    'Norm': 1,
    '>7': 2,
    '>8': 3
}

map_glu = {
    0: 0,
    'None': 0,
    'Norm': 1,
    '>200': 2,
    '>300': 3
}

df['A1Cresult'] = df['A1Cresult'].map(map_a1c)
df['max_glu_serum'] = df['max_glu_serum'].map(map_glu)

categorical_cols = [
    'race', 'age', 'admission_type', 'discharge_disposition',
    'admission_source', 'diag_1_category', 'diag_2_category',
    'diag_3_category', 'medical_specialty'
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

med_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'insulin'
]

med_map = {
    'No': 0,        # not prescribed
    'Down': 1,      # dosage decreased
    'Steady': 2,    # dosage unchanged
    'Up': 3         # dosage increased
}

# Apply mapping
for col in med_cols:
    df[col] = df[col].map(med_map)

# 0=low(≤5), 1=moderate(6-10), 2=high(11-20), 3=very high(>20)
df['polypharmacy_level'] = pd.cut(
    df['num_medications'],
    bins=[0, 5, 10, 20, np.inf],
    labels=[0, 1, 2, 3],
    include_lowest=True
).astype(int)

# 0=low(≤3), 1=moderate(4-6), 2=high(7-9), 3=severe(≥10)
df['comorbidity_score'] = pd.cut(
    df['number_diagnoses'],
    bins=[0, 3, 6, 9, np.inf],
    labels=[0, 1, 2, 3],
    include_lowest=True
).astype(int)

# Total hospital contacts
df['total_visits'] = (
    df['number_inpatient'] + df['number_emergency'] + df['number_outpatient']
)

# Binary flags for hospital use
df['had_inpatient']   = (df['number_inpatient']   > 0).astype(int)
df['had_emergency']   = (df['number_emergency']   > 0).astype(int)
df['had_outpatient']  = (df['number_outpatient']  > 0).astype(int)

# Frequent visitor flag (≥5 contacts)
df['frequent_visitor'] = (df['total_visits'] >= 5).astype(int)

# 0=short(1-3 days), 1=medium(4-6 days), 2=long(7-10 days), 3=very long(>10 days)
df['stay_length_cat'] = pd.cut(
    df['time_in_hospital'],
    bins=[0, 3, 6, 10, np.inf],
    labels=[0, 1, 2, 3],
    include_lowest=True
).astype(int)

# Drop original columns
cols_to_drop = [
    'num_medications', 'number_diagnoses',
    'number_inpatient', 'number_emergency',
    'number_outpatient', 'time_in_hospital'
]
df = df.drop(columns=cols_to_drop)


# continuous variables to scale
to_scale = ['total_visits']

# Initialize scaler
scaler = StandardScaler()

# Fit and transform
df[to_scale] = scaler.fit_transform(df[to_scale])
