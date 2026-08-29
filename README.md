# 🧠 NeuroPath AI

### AI-Driven Prioritization System for Early Alzheimer's Diagnostic Pathways

NeuroPath AI is a machine-learning-based **clinical decision-support prototype** designed to prioritize patients for further Alzheimer's diagnostic evaluation using demographic, cognitive, and MRI-derived clinical features.

Instead of simply predicting a diagnostic label, NeuroPath AI produces a **risk/prioritization score** and assigns patients to **LOW, MEDIUM, or HIGH priority groups**, helping demonstrate how limited diagnostic resources could be allocated more effectively.

---

##  Live Application

###  [Launch NeuroPath AI](https://neuropath-ai.streamlit.app/)

The deployed Streamlit application allows users to enter patient information and receive:

- Model probability
- Risk / prioritization score
- LOW / MEDIUM / HIGH priority classification
- Suggested clinical assessment priority

> ⚠️ **Research Prototype:** NeuroPath AI is intended for research, educational, and clinical decision-support demonstration purposes only. It does not provide a medical diagnosis or a clinically validated probability of Alzheimer's disease.

---

##  Problem Statement

Early detection of Alzheimer's disease is important for timely clinical intervention.

However, advanced diagnostic procedures such as MRI and PET imaging can be expensive, time-consuming, and limited in availability.

In real-world clinical pathways, large numbers of patients may undergo initial cognitive assessment, while only a smaller subset can proceed to advanced diagnostic testing.

This creates an important prioritization problem:

> **Which patients should be evaluated first?**

NeuroPath AI explores a machine-learning approach for supporting this decision by generating a patient-level prioritization score from already available clinical information.

---

## 💡 Proposed Solution

NeuroPath AI follows the concept of a progressive diagnostic pathway:

```text
Patient Clinical Data
        ↓
Data Preprocessing
        ↓
Machine Learning Model
        ↓
Risk / Prioritization Score
        ↓
LOW / MEDIUM / HIGH Priority
        ↓
Clinical Decision Support
```

The system is designed to **support clinicians rather than replace clinical judgment**.

---

##  DATASETS

The current prototype uses the **OASIS-2 Longitudinal MRI Dataset**.

The dataset contains longitudinal information from older adults across multiple clinical visits.

Available information includes:

- Demographic information
- Cognitive assessment scores
- Clinical dementia ratings
- MRI-derived volumetric measurements
- Longitudinal patient visits

### Dataset Overview

- **150 unique subjects**
- **373 total visits**
- Longitudinal follow-up observations
- Nondemented, Demented, and Converted subject trajectories

---

## FEATURES USED

The current V2 model uses seven features:

| Feature | Description |
|---|---|
| Age | Patient age |
| M/F | Biological sex recorded in the dataset |
| EDUC | Years of education |
| SES | Socioeconomic status |
| MMSE | Mini-Mental State Examination score |
| eTIV | Estimated Total Intracranial Volume |
| nWBV | Normalized Whole Brain Volume |

### Features intentionally excluded

**CDR (Clinical Dementia Rating)** is used to construct the V2 target and is therefore **not provided to the model as an input feature**.

Identifiers and redundant/non-modeling variables such as Subject ID, MRI ID, Group, ASF, and Hand are also excluded from the V2 feature set.

---

#  Model Development

## Model V1 — Baseline Classification

The first version established the complete ML workflow using baseline Visit 1 data.

V1 performed:

```text
Patient → Nondemented / Demented
```

Models evaluated:

- Logistic Regression
- Random Forest

Random Forest was selected as the V1 candidate model.

### V1 Test Results

| Metric | Score |
|---|---:|
| Accuracy | 75.0% |
| Precision | 80.0% |
| Recall | 61.5% |
| F1 Score | 69.6% |
| ROC-AUC | 79.2% |

V1 served as the baseline before moving toward the actual patient-prioritization objective.

---

#  Model V2 — Risk Prioritization System

V2 extends the project from simple diagnostic-group classification toward **clinical prioritization**.

The target is based on the patient's current CDR-defined state:

```text
CDR = 0       → Lower-priority state
CDR >= 0.5    → Higher-priority state
```

**CDR itself is never supplied as a model input.**

Therefore, the model output should be interpreted as a probability associated with the prototype's **CDR-defined higher-priority state**, not as a probability that a patient has or will develop Alzheimer's disease.

---

##  Subject-Level Data Splitting

Because OASIS-2 contains multiple visits from the same subjects, randomly splitting individual rows could cause information from one patient to appear in both training and testing data.

To reduce this source of leakage, V2 uses **subject-grouped splitting**.

```text
Training Subjects ─────┐
                       ├── No subject overlap
Testing Subjects ──────┘
```

V2 uses:

- StratifiedGroupKFold
- Subject-level train/test separation
- Group-aware model cross-validation

The held-out split contains:

- **121 training subjects**
- **29 testing subjects**
- **298 training visits**
- **75 testing visits**

---

## ⚙️ Preprocessing Pipeline

Preprocessing is performed using Scikit-learn `Pipeline` and `ColumnTransformer`.

The pipeline includes:

- Median imputation for missing numerical values
- Standard scaling of numerical features
- One-hot encoding of categorical features
- Integrated preprocessing and model inference

Keeping preprocessing inside the pipeline helps reduce data leakage and ensures the same transformations are applied during training and inference.

---

##  Models Compared

Three machine-learning models were evaluated for V2:

- Logistic Regression
- Random Forest
- Gradient Boosting

### Grouped Cross-Validation Performance

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.5% | 73.0% | 76.3% | **89.7%** |
| Random Forest | 77.5% | 72.2% | 73.5% | 85.0% |
| Gradient Boosting | 77.9% | 69.2% | 72.7% | 84.2% |

**Logistic Regression** was selected for the current V2 prototype.

---

## 📈 Held-Out Test Performance

At the standard 0.50 decision threshold:

| Metric | Score |
|---|---:|
| Accuracy | **81.3%** |
| Precision | **95.5%** |
| Recall / Sensitivity | **61.8%** |
| Specificity | **97.6%** |
| F1 Score | **75.0%** |
| ROC-AUC | **87.1%** |
| PR-AUC | **88.8%** |
| Brier Score | **0.147** |

These results are experimental and should not be interpreted as clinical validation.

---

# 🚦 Patient Prioritization

Rather than relying only on binary classification, NeuroPath AI converts the model output into three prototype priority groups.

| Model Score | Priority |
|---|---|
| `< 0.20` | 🟢 LOW |
| `0.20 – < 0.40` | 🟡 MEDIUM |
| `≥ 0.40` | 🔴 HIGH |

These thresholds were derived experimentally from training predictions with an emphasis on sensitivity.

They are **research thresholds and are not clinically validated cutoffs**.

The intended interpretation is:

```text
LOW
↓
Routine follow-up

MEDIUM
↓
Consider prioritizing for further clinical assessment

HIGH
↓
Prioritize for prompt clinical assessment
```

---

# 💻 Streamlit Clinical Decision-Support Dashboard

NeuroPath AI includes an interactive Streamlit interface where users can enter:

```text
Age
Sex
Education
SES
MMSE
eTIV
nWBV
```

The application then performs:

```text
Patient Input
      ↓
Saved Scikit-learn Pipeline
      ↓
Probability
      ↓
Risk / Prioritization Score
      ↓
Priority Classification
      ↓
Decision-Support Recommendation
```


# 🔮 Future Development

NeuroPath AI is designed as a progressive project rather than a finished clinical system.

Planned improvements include:

- Improved probability calibration
- Stronger subject-grouped validation
- Patient-level explainability using SHAP
- Expanded longitudinal progression analysis
- Further analysis of Converted subjects
- External dataset validation
- Raw MRI image processing
- Deep-learning-based MRI feature extraction
- Multimodal data fusion
- More clinically informed prioritization thresholds

A future architecture could extend the system toward:

```text
Cognitive Screening
        ↓
Clinical + Blood Biomarkers
        ↓
MRI Analysis
        ↓
PET / Advanced Testing
```

with patient risk re-evaluated progressively as additional information becomes available.

---

## ⚠️ Limitations

The current prototype has several important limitations:

- It is trained on a relatively small research dataset.
- It has not undergone external clinical validation.
- Priority thresholds are experimental.
- The V2 target represents a CDR-defined higher-priority state rather than future Alzheimer's conversion.
- Current model probabilities should not be interpreted as clinically validated disease probabilities.
- Additional validation and methodological refinement are required before any real-world clinical use.

---


---
