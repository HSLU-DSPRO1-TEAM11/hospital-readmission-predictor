#  Predicting Hospital Readmissions: A Case Study on Diabetic Patients

##  Overview
This project investigates the factors that contribute to hospital readmissions among diabetic patients.  
By analyzing structured healthcare data, we aim to develop a machine learning model that predicts the likelihood of a patient being readmitted after discharge.

Our goals:
1. **Support hospitals** in identifying which treatments or procedures are less effective.
2. **Reduce waiting times** by improving patient flow and preventing avoidable readmissions.

This project is part of the **DSPRO1 (Data Science Project HS25)** module.

---

##  Team Members
- **Taulant Saliu**  
- **Maximilian Jager**  
- **Neela Patil**

---

##  Project Objectives
- Build a reproducible **data pipeline** for importing, cleaning, and transforming hospital data.  
- Explore the dataset through **statistical analysis and visualization**.  
- Develop and evaluate **machine learning models** to predict patient readmission.  
- Interpret results to provide **data-driven insights** for healthcare optimization.

---

##  Data Description
The dataset contains 10 years of hospital records for diabetic patients across multiple U.S. hospitals.  
Each record represents a single patient admission and includes:
- **Demographics:** age, gender, race  
- **Stay details:** time in hospital, number of procedures, lab tests, medications  
- **Diagnoses:** primary and secondary diagnosis codes  
- **Target variable:** `readmitted` (Yes/No)

This dataset is **structured** and tabular, ideal for supervised learning approaches.

---

## Requirements
Install dependencies before running any notebook:
```
pip install -r requirements.txt
```

---

## Reproducibility
All code in this project runs **end-to-end** from the provided Jupyter notebooks and helper Python scripts.  
No manual data manipulation or external credentials are required.

The project is designed to be **fully reproducible**:

- All processing steps are documented and automated.  
- Data loading, cleaning, feature engineering, and model training can be executed in sequence.  
- Folder paths are relative, ensuring the notebooks work on any system.  
- The repository is version-controlled with Git and includes all necessary files for replication.

These practices ensure that anyone who clones the repository can reproduce the same workflow, results, and insights.
