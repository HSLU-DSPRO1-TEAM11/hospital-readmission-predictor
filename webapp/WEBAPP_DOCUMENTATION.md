# Hospital Readmission Risk Assessment Webapp

## Documentation and Design Decisions

This document explains the design decisions, architecture, and implementation details of the Hospital Readmission Risk Assessment webapp.

---

## 1. Project Overview

### What We Built
A **Patient Discharge Risk Assessment Tool** - a web application that allows hospital staff to assess the readmission risk of diabetic patients at discharge time.

### Use Case
When a diabetic patient is being discharged:
1. Clinical staff (nurses, discharge planners, case managers) enter patient data
2. The app predicts the probability of readmission within 30 days
3. Risk level and tailored recommendations guide discharge planning decisions

### Why This Use Case?
- **Proactive intervention**: Identify high-risk patients before discharge
- **Resource allocation**: Focus extra care on patients who need it most
- **Cost reduction**: Hospital readmissions are expensive and penalized by insurance
- **Better outcomes**: Targeted follow-up improves patient health

---

## 2. Technology Decisions

### Why Streamlit?

We chose **Streamlit** as the web framework for the following reasons:

| Consideration | Decision |
|--------------|----------|
| **Language** | Python - matches the existing ML pipeline |
| **Learning curve** | Minimal - simple API for creating web interfaces |
| **ML integration** | Native support for pandas, numpy, scikit-learn models |
| **UI components** | Built-in widgets for forms, charts, metrics |
| **Deployment** | Easy deployment via Streamlit Cloud or any Python host |
| **Development speed** | Rapid prototyping - single file for entire app |

**Alternatives considered:**
- Flask/FastAPI + React: More complex, requires frontend expertise
- Dash: Good alternative, but Streamlit is simpler for demos
- Gradio: Similar to Streamlit, but less customizable

### Why XGBoost as the Model?

Based on the hyperparameter tuning results:

| Model | Accuracy | F1 Score | Recall | ROC AUC |
|-------|----------|----------|--------|---------|
| XGBoost (tuned) | 0.5712 | **0.6593** | **0.9000** | 0.6975 |
| Random Forest (tuned) | 0.6351 | 0.5924 | 0.5752 | 0.6867 |

**Key reasons for choosing XGBoost:**
1. **Highest F1 score (0.66)**: Best balance of precision and recall
2. **Exceptional recall (90%)**: Catches 90% of patients who will be readmitted
3. **Clinical priority**: In healthcare, missing a high-risk patient (false negative) is worse than extra monitoring of low-risk patients (false positive)

---

## 3. Architecture

### File Structure

```
webapp/
├── app.py                  # Main Streamlit application
├── preprocessing.py        # Data transformation functions
├── export_model.py         # Script to export model from notebooks
├── requirements.txt        # Python dependencies
├── WEBAPP_DOCUMENTATION.md # This documentation
└── model/
    ├── best_model.pkl      # Trained XGBoost model
    ├── scaler.pkl          # StandardScaler for numeric features
    └── feature_columns.json # Feature column names in order
```

### Data Flow

```
User Input (Form)
       │
       ▼
┌─────────────────┐
│ preprocessing.py │  ← Transforms raw input to model features
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   XGBoost Model  │  ← Predicts readmission probability
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Risk Assessment │  ← Categorizes risk and generates recommendations
└─────────────────┘
       │
       ▼
User Interface Display
```

---

## 4. Key Implementation Decisions

### 4.1 Feature Preprocessing

**Challenge**: The model expects 180 one-hot encoded features, but users input categorical values.

**Solution**: The `preprocessing.py` module handles the transformation:
1. User selects from dropdowns (e.g., "Emergency" for admission type)
2. We create a zero-initialized DataFrame with all 180 columns
3. We set the appropriate one-hot column to 1 for each categorical input
4. Numeric features are scaled using the saved StandardScaler

**Reasoning**:
- Users don't need to understand one-hot encoding
- The preprocessing exactly matches the training pipeline
- Column order is preserved via `feature_columns.json`

### 4.2 Input Simplification

**Challenge**: The raw data has complex ICD-9 diagnosis codes (e.g., "250.83").

**Solution**: We use diagnosis categories instead:
- Users select from 20 predefined medical categories
- Example: "Diseases of the circulatory system" instead of code "410.00"

**Reasoning**:
- Hospital staff can quickly identify the correct category
- No need for ICD-9 code lookup
- Categories match how the model was trained

### 4.3 Risk Level Thresholds

We defined three risk levels based on probability:

| Probability | Risk Level | Rationale |
|-------------|------------|-----------|
| < 30% | Low Risk | Below average risk, standard care appropriate |
| 30-60% | Medium Risk | Above average, enhanced monitoring recommended |
| > 60% | High Risk | Elevated risk, comprehensive planning needed |

**Reasoning**:
- Thresholds align with clinical actionability
- The 90% recall means most true readmissions will be flagged
- Even "medium risk" patients get enhanced recommendations

### 4.4 Dynamic Recommendations

Recommendations are generated based on:
1. **Risk level**: Higher risk = more intensive interventions
2. **Patient factors**: Specific recommendations for:
   - Multiple prior hospitalizations
   - High comorbidity burden
   - High medication count
   - Frequent healthcare utilization
   - Extended hospital stay

**Reasoning**:
- Generic recommendations are less actionable
- Factor-specific recommendations help target interventions
- Clinical staff can prioritize based on patient context

### 4.5 Model Export Strategy

**Challenge**: The notebooks train models dynamically without saving them portably.

**Solution**: Created `export_model.py` which:
1. Runs the same feature engineering pipeline
2. Applies the same feature selection (top 25 features by importance)
3. Trains XGBoost with the best hyperparameters
4. Saves model, scaler, and feature columns

**Reasoning**:
- Ensures exact reproducibility of training
- Portable artifacts can be deployed anywhere
- Feature column order is preserved

---

## 5. Running the Webapp

### Prerequisites
```bash
# Install dependencies
pip install -r webapp/requirements.txt
```

### First-time Setup (Export Model)
```bash
# Must run from notebooks directory due to relative paths
cd notebooks
python ../webapp/export_model.py
```

### Start the Application
```bash
# From project root
streamlit run webapp/app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 6. User Interface Design

### Input Sections

1. **Demographics**: Gender, age group, race
2. **Admission Details**: Type, source, discharge disposition
3. **Medical Specialty**: Primary treating specialty
4. **Diagnosis Information**: Three diagnosis categories
5. **Clinical Information**: Procedures, medications, lab results
6. **Prior Healthcare Utilization**: Previous visits (inpatient, ER, outpatient)
7. **Medication Details**: Individual diabetes medications (collapsible)

### Output Sections

1. **Risk Score**: Probability (0-100%) with progress bar
2. **Risk Level**: Color-coded badge (green/orange/red)
3. **Recommendations**: Numbered list of actionable items
4. **Patient Summary**: Key metrics at a glance

### Design Principles

- **Progressive disclosure**: Most-used inputs first, detailed options expandable
- **Sensible defaults**: Common values pre-selected
- **Visual clarity**: Color-coding for risk levels
- **Actionable output**: Specific recommendations, not just scores

---

## 7. Model Performance Context

### What the Scores Mean

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 57% | Predicts correctly 57% of the time |
| Precision | 52% | Of predicted readmissions, 52% actually occur |
| **Recall** | **90%** | Catches 90% of actual readmissions |
| F1 Score | 66% | Balance of precision and recall |
| ROC AUC | 70% | Good discrimination ability |

### Clinical Interpretation

- **High recall is critical**: We catch 9 out of 10 patients who will be readmitted
- **Trade-off**: Some false alarms (48% of flagged patients won't be readmitted)
- **Acceptable in healthcare**: Extra monitoring is less costly than missed readmissions

### Limitations

1. **Historical data (1999-2008)**: Clinical practices have evolved
2. **Single dataset**: May not generalize to all hospital systems
3. **Missing context**: Cannot capture all clinical nuances
4. **Demonstration tool**: Should not replace clinical judgment

---

## 8. Future Improvements

If this were a production system, consider:

1. **SHAP explanations**: Show which factors most influenced the prediction
2. **Patient history lookup**: Connect to EHR for auto-population
3. **Audit logging**: Track predictions for model monitoring
4. **Threshold tuning**: Allow hospitals to adjust sensitivity
5. **A/B testing**: Compare outcomes with/without tool usage
6. **Retraining pipeline**: Update model with recent data

---

## 9. Summary

This webapp demonstrates how the hospital readmission prediction model could be used in a clinical setting. Key design decisions prioritized:

- **Usability**: Simple interface for busy clinical staff
- **Clinical relevance**: Actionable recommendations, not just scores
- **Safety**: High recall to minimize missed high-risk patients
- **Transparency**: Clear documentation of limitations

The tool is meant for demonstration purposes and should complement, not replace, clinical judgment.

---

*HSLU DSPRO1 - Hospital Readmission Prediction Project*
*December 2024*
