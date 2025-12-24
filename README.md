# AI Impact on Jobs (2030)
## Risk Prediction, Job Resilience & Hybrid Recommender System

This project was developed by **Faruk** and **Ahmetsefa** as part of the **Introduction to Machine Learning** course.
The aim of the project is to analyze how artificial intelligence may affect different jobs by predicting AI risk, classifying job exposure levels, recommending future-proof alternatives, and calculating job resilience to AI.

---

## Project Overview

Instead of relying on a single model, the project follows a multi-stage machine learning pipeline:

- Predicting AI impact on jobs using **regression**
- Categorizing jobs into **risk classes** using classification
- Recommending **lower-risk alternative jobs** with a hybrid recommender system
- Computing an **AI resilience score** for each job

The focus is not only on which jobs are at risk, but also on which jobs are more adaptable and resilient in the AI era.

---

## Dataset

**File:** `My_Data.csv`

Main columns:
- `AI Impact` (percentage format, e.g. `"65%"`)
- `Tasks`
- `AI models`
- `AI_Workload_Ratio`
- `Domain`
- `Job titiles` (column name kept as in the original dataset)

### Preprocessing Steps
- Removed `%` symbol and converted AI Impact to numeric
- Handled infinite values
- Filled missing values using median where applicable
- Clipped `AI_Workload_Ratio` to the range `[0, 1]`

---

## Methods and Models

### 1. AI Impact Regression
**Model:** RandomForestRegressor

The regression model predicts the AI Impact percentage for each job.

- Numerical features: `Tasks`, `AI models`, `AI_Workload_Ratio`
- Categorical feature: `Domain` (One-Hot Encoded)
- StandardScaler used for numerical features

**Evaluation metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Actual vs predicted scatter plot

---

### 2. Risk Classification
**Model:** RandomForestClassifier

Jobs are grouped into AI risk categories based on the intensity of expected AI-driven automation:

- **Low Risk:** Roles where AI impact is limited and mainly supportive  
- **Medium Risk:** Roles likely to be partially automated or significantly transformed  
- **High Risk:** Roles with a high likelihood of extensive automation or replacement  



The classifier uses both numerical and categorical features:
- Numerical: `Tasks`, `AI models`, `AI_Workload_Ratio`
- Categorical: `Job titiles`, `Domain`

A preprocessing pipeline is applied, including feature scaling for numerical variables and one-hot encoding for categorical variables, followed by a Random Forest classifier.

The purpose of this step is to move from continuous risk estimation to **interpretable risk groups**, making the results easier to analyze and compare across jobs.

**Output file:**  
`model_risk_predictions.csv`

---

### 3. Hybrid Recommender System (Future-Proof Jobs)

For jobs classified as **high risk**, a recommender system suggests alternative jobs with lower AI risk.

**Techniques used:**
- Sentence embeddings using **SentenceTransformer** (`all-MiniLM-L6-v2`)
- Cosine similarity for semantic similarity
- Domain similarity using One-Hot Encoding
- Numeric feature similarity using scaled features

**Hybrid similarity weighting:**
- 45% Job title similarity
- 35% Domain similarity
- 20% Numeric feature similarity

**Output file:**  
`hybrid_recommendations_offline.csv`

---

### 4. Job Resilience Index

The Job Resilience Index is designed to quantify how well a job can withstand AI-driven automation over time.

It is derived from the predicted AI Impact score, where lower AI impact implies higher resilience.  
Rather than focusing solely on automation risk, this index highlights roles that are more adaptable, flexible, and less susceptible to full automation.

The score enables meaningful comparisons across jobs and domains, helping identify roles that are more likely to remain relevant in the long term.

**Output file:**  
`job_model_resilience_index.csv`
---

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn sentence-transformers
```

---

## How to Run

```bash
python 01_eda.py
python 02_risk_regression.py
python 03_risk_classification.py
python 04_hybrid_recommender.py
python 05_resilience_index.py
```

---

## Outputs

- `model_risk_predictions.csv`
- `hybrid_recommendations_offline.csv`
- `job_model_resilience_index.csv`

---


