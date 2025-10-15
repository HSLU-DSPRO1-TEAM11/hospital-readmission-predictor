# 📘 Dataset Description – Hospital Readmissions (Diabetic Patients)

## Overview
This dataset contains medical records of diabetic patients collected from multiple U.S. hospitals over a ten-year period.  
Each record represents a single hospital **encounter (admission)** for a patient and includes demographic information, diagnoses, procedures, laboratory tests, prescribed medications, and readmission status.

---

## 📄 Data Information

- **Number of Rows:** 101,766  
- **Number of Columns:** 50  
- **File Type:** CSV  
- **Encoding:** UTF-8  

Each row corresponds to one hospital encounter, and each patient can have multiple encounters recorded.

---

## 📂 Data Structure

| Variable Name | Role | Type | Description | Units | Missing Values |
|----------------|------|------|--------------|--------|----------------|
| **encounter_id** | ID | – | Unique identifier of an encounter. | – | No |
| **patient_nbr** | ID | – | Unique identifier of a patient. | – | No |
| **race** | Feature | Categorical | Patient’s race. Values: *Caucasian, Asian, African American, Hispanic, Other.* | – | Yes |
| **gender** | Feature | Categorical | Patient’s gender. Values: *Male, Female, Unknown/Invalid.* | – | No |
| **age** | Feature | Categorical | Age group in 10-year intervals: [0–10), [10–20), …, [90–100). | – | No |
| **weight** | Feature | Categorical | Patient’s weight in pounds. | Pounds | Yes |
| **admission_type_id** | Feature | Categorical | Admission type (e.g., emergency, urgent, elective, newborn, not available). | – | No |
| **discharge_disposition_id** | Feature | Categorical | Discharge destination (e.g., home, expired, not available). | – | No |
| **admission_source_id** | Feature | Categorical | Source of admission (e.g., physician referral, emergency room, transfer). | – | No |
| **time_in_hospital** | Feature | Integer | Duration of hospital stay. | Days | No |
| **payer_code** | Feature | Categorical | Payment source (e.g., Blue Cross/Blue Shield, Medicare, self-pay). | – | Yes |
| **medical_specialty** | Feature | Categorical | Specialty of the admitting physician (84 distinct values). | – | Yes |
| **num_lab_procedures** | Feature | Integer | Number of lab tests performed during the encounter. | – | No |
| **num_procedures** | Feature | Integer | Number of procedures (other than lab tests) performed during the encounter. | – | No |
| **num_medications** | Feature | Integer | Number of distinct generic medications administered. | – | No |
| **number_outpatient** | Feature | Integer | Number of outpatient visits in the year before the encounter. | – | No |
| **number_emergency** | Feature | Integer | Number of emergency visits in the year before the encounter. | – | No |
| **number_inpatient** | Feature | Integer | Number of inpatient visits in the year before the encounter. | – | No |
| **diag_1** | Feature | Categorical | Primary diagnosis (first 3 digits of ICD9). | – | Yes |
| **diag_2** | Feature | Categorical | Secondary diagnosis (first 3 digits of ICD9). | – | Yes |
| **diag_3** | Feature | Categorical | Additional secondary diagnosis (first 3 digits of ICD9). | – | Yes |
| **number_diagnoses** | Feature | Integer | Total number of diagnoses recorded. | – | No |
| **max_glu_serum** | Feature | Categorical | Glucose serum test result: *>200, >300, normal, none.* | – | No |
| **A1Cresult** | Feature | Categorical | A1C test result: *>8, >7, normal, none.* | – | No |
| **medication features** | Feature | Categorical | For each drug (e.g., *metformin, insulin, glipizide*), indicates dosage change: *up, down, steady, no.* | – | No |
| **change** | Feature | Categorical | Indicates if there was a change in diabetic medications. Values: *change, no change.* | – | No |
| **diabetesMed** | Feature | Categorical | Indicates whether any diabetic medication was prescribed. Values: *yes, no.* | – | No |
| **readmitted** | Target | Categorical | Readmission status: *<30* (within 30 days), *>30* (after 30 days), *No* (no readmission). | – | No |

---

## 🧬 Feature Groups

- **Identification:**  
  `encounter_id`, `patient_nbr`

- **Demographics:**  
  `race`, `gender`, `age`, `weight`

- **Hospitalization Details:**  
  `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `time_in_hospital`, `payer_code`, `medical_specialty`

- **Historical Visits:**  
  `number_outpatient`, `number_emergency`, `number_inpatient`

- **Diagnoses & Tests:**  
  `diag_1`, `diag_2`, `diag_3`, `number_diagnoses`, `num_lab_procedures`, `num_procedures`

- **Medications (22 drugs):**  
  Includes: `metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`,  
  `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `examide`, `citoglipton`,  
  `insulin`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`

- **Lab Results:**  
  `max_glu_serum`, `A1Cresult`

- **Medication Changes:**  
  `change`, `diabetesMed`

- **Target Variable:**  
  `readmitted`

---

## ⚙️ Data Notes

- The dataset includes both **categorical** and **numerical** variables.  
- Several columns contain **missing values**, notably `race`, `weight`, `payer_code`, `medical_specialty`, and the diagnosis fields.  
- Medication-related variables represent both prescription presence and dosage changes.  
- Patients can appear multiple times if they have several encounters recorded.  
- The target variable `readmitted` represents whether a patient was readmitted within 30 days, after 30 days, or not at all.
