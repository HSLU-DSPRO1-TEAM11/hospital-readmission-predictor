# Predicting Hospital Readmissions: A Case Study on Diabetic Patients

## Overview
This project investigates the factors that contribute to hospital readmissions among diabetic patients.
By analyzing structured healthcare data, we aim to develop a machine learning model that predicts the likelihood of a patient being readmitted after discharge.

Our goals:
1. **Support hospitals** in identifying which treatments or procedures are less effective.
2. **Reduce waiting times** by improving patient flow and preventing avoidable readmissions.

This project is part of the **DSPRO1 (Data Science Project HS25)** module.

---

## Team Members
- **Taulant Saliu**
- **Maximilian Jager**
- **Neela Patil**

---

## Project Objectives
- Build a reproducible **data pipeline** for importing, cleaning, and transforming hospital data.
- Explore the dataset through **statistical analysis and visualization**.
- Develop and evaluate **machine learning models** to predict patient readmission.
- Interpret results to provide **data-driven insights** for healthcare optimization.

---

## Results Summary

The tuned XGBoost binary classifier achieved the best performance:

| Metric | Value |
|--------|-------|
| Accuracy | 57.1% |
| Precision | 52.0% |
| Recall | 90.0% |
| F1-Score | 0.66 |
| ROC AUC | 0.70 |

**Key Finding:** Prior healthcare utilization (previous inpatient admissions and total visits) accounts for over 20% of the model's predictive power.

---

## Data Description
The dataset contains 10 years of hospital records for diabetic patients across multiple U.S. hospitals.
Each record represents a single patient admission and includes:
- **Demographics:** age, gender, race
- **Stay details:** time in hospital, number of procedures, lab tests, medications
- **Diagnoses:** primary and secondary diagnosis codes (ICD-9)
- **Target variable:** `readmitted`
  - **Binary classification:** Yes/No (readmitted or not)
  - **3-class classification:** NO (not readmitted), >30 (readmitted after 30 days), <30 (readmitted within 30 days)

This dataset is **structured** and tabular, ideal for supervised learning approaches.

---

## Project Structure

```
hospital-readmission-predictor/
├── notebooks/                     # Jupyter notebooks (run in order)
│   ├── data_exploration.ipynb     # Initial EDA and visualization
│   ├── data_cleaning.ipynb        # Binary classification data cleaning
│   ├── feature_engineering.ipynb  # Feature creation and selection
│   ├── machine_learning.ipynb     # Model training with MLflow tracking
│   ├── hyperparameter_tuning.ipynb# Model optimization
│   └── multiclass_baseline.ipynb  # 3-class classification approach
├── src/                           # Reusable Python modules
│   ├── data_cleaning.py           # Binary classification cleaning
│   ├── data_cleaning_multiclass.py# 3-class classification cleaning
│   ├── feature_engineering.py     # Binary feature engineering
│   └── feature_engineering_multiclass.py # 3-class feature engineering
├── data/
│   ├── raw/                       # Original dataset files
│   └── processed/                 # Generated processed data
├── webapp/                        # Streamlit web application
├── mlruns/                        # MLflow run metadata
└── mlartifacts/                   # MLflow model artifacts
```

---

## Prerequisites

- **Python 3.9+**
- **MLflow** (for experiment tracking - must be running before ML notebooks)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HSLU-DSPRO1-TEAM11/hospital-readmission-predictor.git
   cd hospital-readmission-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Reproducing Results

### Step 1: Start MLflow Tracking Server

Before running the machine learning notebooks, start the MLflow server:

```bash
mlflow ui --port 5000
```

Keep this running in a separate terminal. Access the UI at http://localhost:5000

### Step 2: Run Notebooks in Order

Navigate to the `notebooks/` directory and execute the notebooks in this sequence:

| Order | Notebook | Description |
|-------|----------|-------------|
| 1 | `data_exploration.ipynb` | Exploratory data analysis and visualization |
| 2 | `data_cleaning.ipynb` | Data cleaning for binary classification |
| 3 | `feature_engineering.ipynb` | Feature creation, encoding, and selection |
| 4 | `machine_learning.ipynb` | Train baseline models (LogisticRegression, XGBoost, DecisionTree, RandomForest) |
| 5 | `hyperparameter_tuning.ipynb` | Optimize model hyperparameters |
| 6 | `multiclass_baseline.ipynb` | *(Optional)* 3-class classification approach |

### Step 3: View Results in MLflow

After running the ML notebooks, view experiment results at http://localhost:5000

**Tracked experiments:**
- `model-comparison` - Binary classification baseline results
- `hyperparameter-tuning` - Optimized model results
- `multiclass-baseline` - 3-class classification results

**Logged metrics:** Accuracy, F1 Score, Precision, Recall, ROC AUC

---

## Running the Web Application

A Streamlit web app is available for interactive predictions:

```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

The app provides:
- Real-time readmission risk prediction
- Risk level categorization (High/Medium/Low)
- Clinical recommendations based on patient data

---

## Data Source and Acknowledgements
The dataset used in this project originates from the  
**UCI Machine Learning Repository**:  
[Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)


Acknowledgements:
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.
