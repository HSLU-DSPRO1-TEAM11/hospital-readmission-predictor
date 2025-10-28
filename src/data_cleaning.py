import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_data():
    """Load and clean the diabetic_data dataset, returning a cleaned pandas DataFrame."""
    #Load data
    df_data = pd.read_csv("../data/raw/diabetic_data.csv")
    df_mapping = pd.read_csv("../data/raw/IDS_mapping.csv", header=None, names=["C1", "C2"])

    df_data.drop('encounter_id', axis=1, inplace=True)

    #df_data.drop('race', axis=1, inplace=True)

    df_data = df_data[df_data['gender'] != 'Unknown/Invalid'] # remove Unknown
    df_data.drop('weight', axis=1, inplace=True) # drop weight as not enough values
    df_data = df_data[df_data['diag_1'] != '?'] # drop those 21 rows with unknown first diagnose
    df_data[['diag_2', 'diag_3']] = df_data[['diag_2', 'diag_3']].replace('?', 'Unknown') # replace ? by unknown
    df_data.drop('tolbutamide', axis=1, inplace=True) # only No as value, doesn't help with prediction
    df_data.drop('examide', axis=1, inplace=True)
    df_data.drop('citoglipton', axis=1, inplace=True)
    # impact of these columns to low, 706 of 100'000, less than 1%
    df_data.drop(
        ['glyburide-metformin', 'glipizide-metformin',
         'glimepiride-pioglitazone', 'metformin-rosiglitazone',
         'metformin-pioglitazone'],
        axis=1,
        inplace=True
    )
    df_data['change'] = df_data['change'].map({'No': False, 'Ch': True})  # switch from Ch and No to true and false
    df_data['readmitted'] = df_data['readmitted'].apply(lambda
                                                            x: False if x == 'NO' else True)  # for now only true and false, later maybe differ between less or more than 30 days
    df_data['max_glu_serum'].fillna(0, inplace=True)
    df_data['A1Cresult'].fillna(0, inplace=True)

    # replace the id's with the description
    # first divide the mappings in different df
    start_admission = df_mapping.index[df_mapping.iloc[:, 0] == 'admission_type_id'][0]
    start_discharge = df_mapping.index[df_mapping.iloc[:, 0] == 'discharge_disposition_id'][0]
    start_source = df_mapping.index[df_mapping.iloc[:, 0] == 'admission_source_id'][0]

    df_admission_type = df_mapping.iloc[start_admission + 1: start_discharge]
    df_discharge_disposition = df_mapping.iloc[start_discharge + 1: start_source]
    df_admission_source = df_mapping.iloc[start_source + 1:]

    df_admission_type.columns = ['admission_type_id', 'description']
    df_discharge_disposition.columns = ['discharge_disposition_id', 'description']
    df_admission_source.columns = ['admission_source_id', 'description']

    df_admission_type.drop(index=6, inplace=True)
    df_admission_type.drop(index=9, inplace=True)

    df_discharge_disposition.drop(index=28, inplace=True)
    df_discharge_disposition.drop(index=41, inplace=True)

    df_admission_source.drop(index=58, inplace=True)

    # replace the id's in the original dataframe with the description
    # first make numeric values out of the id's
    df_admission_type['admission_type_id'] = pd.to_numeric(df_admission_type['admission_type_id'])
    df_discharge_disposition['discharge_disposition_id'] = pd.to_numeric(
        df_discharge_disposition['discharge_disposition_id'])
    df_admission_source['admission_source_id'] = pd.to_numeric(df_admission_source['admission_source_id'])
    # second put them in a dictionary for easier work later
    map_admission = dict(zip(df_admission_type['admission_type_id'], df_admission_type['description']))
    map_discharge = dict(
        zip(df_discharge_disposition['discharge_disposition_id'], df_discharge_disposition['description']))
    map_source = dict(zip(df_admission_source['admission_source_id'], df_admission_source['description']))

    # replace the id's with description
    df_data['admission_type_id'] = df_data['admission_type_id'].map(map_admission)
    df_data['discharge_disposition_id'] = df_data['discharge_disposition_id'].map(map_discharge)
    df_data['admission_source_id'] = df_data['admission_source_id'].map(map_source)

    # Replace NaN with 'Not Available'
    df_data['admission_type_id'] = df_data['admission_type_id'].fillna('Not Available')

    # Rename the column
    df_data.rename(columns={'admission_type_id': 'admission_type'}, inplace=True)

    # Replace NaN with 'Unknown/Invalid'
    df_data['discharge_disposition_id'] = df_data['discharge_disposition_id'].fillna('Unknown/Invalid')

    # Rename the column
    df_data.rename(columns={'discharge_disposition_id': 'discharge_disposition'}, inplace=True)

    # Replace NaN with 'Not Available'
    df_data['admission_source_id'] = df_data['admission_source_id'].fillna('Not Available')

    # Rename the column
    df_data.rename(columns={'admission_source_id': 'admission_source'}, inplace=True)

    df_data.drop(columns=['miglitol'], inplace=True)  # only No values
    df_data.drop(columns=['troglitazone'], inplace=True)  # only No values
    df_data.drop(columns=['tolazamide'], inplace=True)  # only No values
    df_data.drop(columns=['acetohexamide'], inplace=True)  # only No values

    return df_data

# If script is executed directly, preview the cleaned dataset
if __name__ == "__main__":
    df_clean = clean_data()
    print(df_clean.info())
    print(df_clean.head())