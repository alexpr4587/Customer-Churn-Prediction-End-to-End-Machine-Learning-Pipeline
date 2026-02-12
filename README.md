# Customer Churn Prediction — Production-Oriented ML System

![Churn App Interface](./assets/churn_app.png)

## Overview

This project implements a complete, production-aware machine learning system for customer churn prediction.  
Rather than focusing solely on model training inside a notebook, the objective was to design, validate, and deploy a reproducible end-to-end pipeline that reflects real-world data science workflows.

The system covers:

- Structured data preprocessing
- Rigorous model validation
- Business-oriented decision threshold tuning
- Model interpretability analysis
- API-based inference
- Containerized deployment

The final result is a fully operational churn prediction application accessible through a web interface backed by a FastAPI inference layer.

---

## Problem Framing

Customer churn prediction is a classical binary classification problem where the goal is to estimate the probability that a customer will discontinue a service.

From a business perspective, predicting churn is only useful if:

- The model is properly validated.
- The probability threshold aligns with retention strategy costs.
- False positives and false negatives are understood and quantified.

This project was developed with those constraints in mind.

---

## Data & Preprocessing

The dataset contains customer-level behavioral and subscription attributes such as:

- Demographics
- Contract type
- Tenure
- Services subscribed
- Billing structure

A structured preprocessing pipeline was implemented using scikit-learn, including:

- Categorical encoding
- Feature scaling where appropriate
- Consistent transformation between training and inference

All preprocessing steps are encapsulated within a single pipeline object to prevent data leakage and ensure reproducibility.

---

## Exploratory Analysis

Exploratory analysis was used as a decision-support tool rather than a purely descriptive exercise. The focus was placed on identifying patterns that could influence modeling choices, such as differences in churn rates across customer segments and the behavior of numerical variables conditioned on churn. Multicollinearity was explicitly checked using Variance Inflation Factor (VIF). These checks were not treated as hard constraints, but as signals to avoid unstable or redundant features that could negatively affect generalization.Insights from this stage directly informed both feature selection and model choice.

---

## Modeling Strategy

The modeling phase followed a disciplined validation framework:

1. A strict train / validation / test split was applied.
2. Cross-validation was used during model selection.
3. Multiple models were compared against a baseline classifier.
4. Hyperparameter tuning was performed using validation performance.
5. The final model was evaluated on a completely unseen test set.

Evaluation metrics included:

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Threshold Optimization

Instead of using the default 0.5 probability cutoff, the decision threshold was optimized based on precision–recall tradeoffs.

This reflects real-world churn management scenarios, where:

- False negatives may result in lost customers.
- False positives may trigger unnecessary retention incentives.

The chosen threshold balances operational cost and recall performance.

---

## Model Interpretation & Error Analysis

Feature importance analysis was conducted to understand the primary drivers of churn risk.

Additionally, structured error analysis was performed to:

- Identify systematic misclassification patterns.
- Evaluate model behavior across tenure segments.
- Inspect probability calibration behavior.

This step ensures the model is not treated as a black box.

---

## Deployment Architecture

The trained pipeline was serialized and deployed through a production-style architecture:

- **FastAPI** serves the model for real-time inference.
- **Streamlit** provides an interactive frontend interface.
- **Docker** ensures environment reproducibility.
- The application runs inside an isolated private network for controlled access.

Architecture Flow:

User Input → Streamlit UI → FastAPI Endpoint → Preprocessing Pipeline → Model → Probability Output

---

## Project Structure

```
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── inference.py
├── api/
│   ├── main.py
├── app/
│   ├── streamlit_app.py
├── models/
│   ├── final_model.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Final Notes

This project was designed to go beyond notebook experimentation.  
The focus was on building a system that reflects production-oriented machine learning engineering principles:

- Reproducibility  
- Clear separation between training and serving  
- Evaluation rigor  
- Deployment awareness  

The goal was not only to build a predictive model, but to design a structured, deployable machine learning solution.
