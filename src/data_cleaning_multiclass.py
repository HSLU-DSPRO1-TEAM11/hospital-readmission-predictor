"""
Data cleaning module for 3-class classification.
Target: readmitted with 3 classes (NO, >30, <30)

This is a modified version of data_cleaning.py that preserves the 3-class
structure instead of converting to binary classification.
"""
import pandas as pd


def clean_data_multiclass():
    """
    Load and clean the diabetic_data dataset for 3-class classification.

    Target encoding:
        0 = NO (not readmitted)
        1 = >30 (readmitted after 30 days)
        2 = <30 (readmitted within 30 days)

    Returns:
        pd.DataFrame: Cleaned dataframe with 3-class target
    """
    # Load data
    df_data = pd.read_csv("../data/raw/diabetic_data.csv")
    df_mapping = pd.read_csv("../data/raw/IDS_mapping.csv", header=None, names=["C1", "C2"])

    # Drop encounter_id (not needed for prediction)
    df_data.drop('encounter_id', axis=1, inplace=True)

    # Remove invalid gender entries
    df_data = df_data[df_data['gender'] != 'Unknown/Invalid']

    # Drop weight column (too many missing values)
    df_data.drop('weight', axis=1, inplace=True)

    # Handle diagnosis columns
    df_data = df_data[df_data['diag_1'] != '?']  # drop rows with unknown primary diagnosis
    df_data[['diag_2', 'diag_3']] = df_data[['diag_2', 'diag_3']].replace('?', 'Unknown')

    # Drop medication columns with only one value (no predictive power)
    df_data.drop('tolbutamide', axis=1, inplace=True)
    df_data.drop('examide', axis=1, inplace=True)
    df_data.drop('citoglipton', axis=1, inplace=True)

    # Drop combination medications with low occurrence (<1%)
    df_data.drop(
        ['glyburide-metformin', 'glipizide-metformin',
         'glimepiride-pioglitazone', 'metformin-rosiglitazone',
         'metformin-pioglitazone'],
        axis=1,
        inplace=True
    )

    # Convert change column to boolean
    df_data['change'] = df_data['change'].map({'No': False, 'Ch': True})

    # 3-CLASS TARGET ENCODING (key difference from binary version)
    # 0 = NO, 1 = >30, 2 = <30
    readmitted_map = {
        'NO': 0,   # not readmitted
        '>30': 1,  # readmitted after 30 days
        '<30': 2   # readmitted within 30 days (most critical)
    }
    df_data['readmitted'] = df_data['readmitted'].map(readmitted_map)

    # Fill missing lab results with 0 (indicates test not performed)
    df_data['max_glu_serum'].fillna(0, inplace=True)
    df_data['A1Cresult'].fillna(0, inplace=True)

    # Map admission/discharge/source IDs to descriptions
    start_admission = df_mapping.index[df_mapping.iloc[:, 0] == 'admission_type_id'][0]
    start_discharge = df_mapping.index[df_mapping.iloc[:, 0] == 'discharge_disposition_id'][0]
    start_source = df_mapping.index[df_mapping.iloc[:, 0] == 'admission_source_id'][0]

    df_admission_type = df_mapping.iloc[start_admission + 1: start_discharge]
    df_discharge_disposition = df_mapping.iloc[start_discharge + 1: start_source]
    df_admission_source = df_mapping.iloc[start_source + 1:]

    df_admission_type.columns = ['admission_type_id', 'description']
    df_discharge_disposition.columns = ['discharge_disposition_id', 'description']
    df_admission_source.columns = ['admission_source_id', 'description']

    # Drop invalid/unknown mapping rows
    df_admission_type.drop(index=6, inplace=True)
    df_admission_type.drop(index=9, inplace=True)
    df_discharge_disposition.drop(index=28, inplace=True)
    df_discharge_disposition.drop(index=41, inplace=True)
    df_admission_source.drop(index=58, inplace=True)

    # Convert to numeric and create mapping dictionaries
    df_admission_type['admission_type_id'] = pd.to_numeric(df_admission_type['admission_type_id'])
    df_discharge_disposition['discharge_disposition_id'] = pd.to_numeric(
        df_discharge_disposition['discharge_disposition_id'])
    df_admission_source['admission_source_id'] = pd.to_numeric(df_admission_source['admission_source_id'])

    map_admission = dict(zip(df_admission_type['admission_type_id'], df_admission_type['description']))
    map_discharge = dict(zip(df_discharge_disposition['discharge_disposition_id'],
                             df_discharge_disposition['description']))
    map_source = dict(zip(df_admission_source['admission_source_id'], df_admission_source['description']))

    # Apply mappings
    df_data['admission_type_id'] = df_data['admission_type_id'].map(map_admission)
    df_data['discharge_disposition_id'] = df_data['discharge_disposition_id'].map(map_discharge)
    df_data['admission_source_id'] = df_data['admission_source_id'].map(map_source)

    # Fill NaN and rename columns
    df_data['admission_type_id'] = df_data['admission_type_id'].fillna('Not Available')
    df_data.rename(columns={'admission_type_id': 'admission_type'}, inplace=True)

    df_data['discharge_disposition_id'] = df_data['discharge_disposition_id'].fillna('Unknown/Invalid')
    df_data.rename(columns={'discharge_disposition_id': 'discharge_disposition'}, inplace=True)

    df_data['admission_source_id'] = df_data['admission_source_id'].fillna('Not Available')
    df_data.rename(columns={'admission_source_id': 'admission_source'}, inplace=True)

    # Drop more columns with no predictive value
    df_data.drop(columns=['miglitol'], inplace=True)
    df_data.drop(columns=['troglitazone'], inplace=True)
    df_data.drop(columns=['tolazamide'], inplace=True)
    df_data.drop(columns=['acetohexamide'], inplace=True)
    df_data.drop(columns=['patient_nbr'], inplace=True)
    df_data.drop(columns=['payer_code'], inplace=True)

    return df_data
