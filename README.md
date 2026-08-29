# 🧠 NeuroPath AI

**AI-Driven Prioritization System for Early Alzheimer's Diagnostic Pathways**

NeuroPath AI is a machine-learning-based clinical decision-support prototype designed to prioritize patients for further Alzheimer's diagnostic evaluation using demographic, cognitive, and MRI-derived features.

### 🚀 Live Application

👉 **[Launch NeuroPath AI](https://neuropath-ai.streamlit.app/)**

> ⚠️ **Research Prototype:** NeuroPath AI is intended for research and clinical decision-support demonstration only. It does not provide a medical diagnosis or a clinically validated probability of Alzheimer's disease.



# NeuroPath AI

AI-driven clinical decision-support system for prioritizing patients in early Alzheimer's diagnostic pathways.

## Problem Statement

Early detection of Alzheimer's disease is difficult because advanced diagnostic resources such as MRI and PET scans are limited.

NeuroPath AI aims to help identify higher-risk patients who should be prioritized for further diagnostic evaluation.

## Dataset

This project currently uses the OASIS-2 Longitudinal MRI dataset.

For Model V1:
- Baseline Visit 1 only
- 136 patients used
- Nondemented and Demented groups
- Converted patients temporarily excluded

## Features Used

- Age
- Sex
- Education
- Socioeconomic Status
- MMSE
- Estimated Total Intracranial Volume (eTIV)
- Normalized Whole Brain Volume (nWBV)

CDR was excluded from V1 to reduce target leakage.

## Preprocessing

The preprocessing pipeline includes:

- Median imputation for missing numerical values
- Standard scaling for numerical features
- One-hot encoding for categorical features

Scikit-learn Pipeline and ColumnTransformer are used to prevent data leakage.

## Models Tested

- Logistic Regression
- Random Forest

Models were compared using 5-fold stratified cross-validation.

## Model V1 Results

Selected baseline model: Random Forest

| Metric | Score |
|---|---|
| Accuracy | 75.0% |
| Precision | 80.0% |
| Recall | 61.5% |
| F1 Score | 69.6% |
| ROC-AUC | 79.2% |

Confusion Matrix:

[[13, 2],
 [ 5, 8]]

## Current Project Status

Model V1 is complete.

The current model performs binary classification:

Patient → Nondemented / Demented

## Next Steps

Model V2 will move toward the actual clinical prioritization objective:

Patient Data
→ Risk Probability
→ Priority Score
→ Low / Medium / High Priority
→ Recommended Next Diagnostic Step

Future work will include:
- Risk prioritization
- Explainable AI using SHAP
- Longitudinal progression analysis
- Converted-patient analysis
- MRI-based deep learning
- Multimodal modeling
- Streamlit dashboard
- Deployment

## Disclaimer

This project is for research and educational purposes only and is not intended to replace professional medical diagnosis.
