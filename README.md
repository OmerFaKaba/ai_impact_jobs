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

Jobs are categorized into three AI risk levels:
- **Low:** 0–50
- **Medium:** 50–70
- **High:** 70–100

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

Using predicted AI Impact values:

**Job_Resilience_Score = 100 − Predicted_AI_Impact**

This score represents how resistant a job is to AI-driven automation.

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


