## Customer Churn Prediction – End-to-End Machine Learning Prooject

## Overview

This project implements an end-to-end machine learning pipeline for customer churn prediction, with an emphasis on **modeling decisions, validation discipline, and production readiness** rather than raw metric optimization.

The goal was to approach this problem as it would be handled in a real data science or machine learning team: starting from exploratory analysis, moving through iterative modeling, and finishing with a fully encapsulated pipeline that could be safely deployed without training–serving inconsistencies.

---

## Problem Context

Customer churn is a high-impact business problem in subscription-based and service-driven industries. Even small improvements in churn prediction can translate into significant revenue retention when paired with targeted interventions.

The task is formulated as a binary classification problem with a clearly imbalanced target. Rather than attempting to “balance away” this imbalance, the modeling strategy explicitly accounts for it during evaluation and threshold selection.

---

## Data Understanding and Preparation

The initial phase focused on understanding the structure and quality of the dataset. This included examining the target distribution, separating numerical and categorical features, and validating assumptions about missing values and data types.

Data cleaning was intentionally conservative. Columns were standardized and normalized where necessary, categorical values were cleaned for consistency, and non-informative or leakage-prone features were removed. Each transformation was applied with the constraint that it must be reproducible at inference time.

The objective at this stage was not to aggressively engineer features, but to ensure that the downstream model would learn from clean, well-defined inputs.

---

## Exploratory Analysis

Exploratory analysis was used as a decision-support tool rather than a purely descriptive exercise. The focus was placed on identifying patterns that could influence modeling choices, such as differences in churn rates across customer segments and the behavior of numerical variables conditioned on churn.

Multicollinearity was explicitly checked using Variance Inflation Factor (VIF). These checks were not treated as hard constraints, but as signals to avoid unstable or redundant features that could negatively affect generalization.

Insights from this stage directly informed both feature selection and model choice.

---

## Feature Engineering

Feature engineering was implemented using custom, scikit-learn compatible transformers. This design choice ensures that all transformations are part of a single, consistent pipeline and that no training-only logic leaks into inference.

Derived features capture aggregated usage patterns and customer behavior signals that are not directly observable from raw columns. Feature removal is also handled explicitly within the pipeline, rather than through ad-hoc dataframe manipulation.

All transformations are deterministic, fitted exclusively on training data, and safe to apply to unseen samples.

---

## Modeling Approach

Multiple models were evaluated during experimentation, with selection driven by stability, interpretability, and probability quality rather than marginal performance gains.

Instead of optimizing exclusively for a single metric, evaluation focused on understanding trade-offs between recall and precision under different decision thresholds. This framing reflects the real business cost of churn prediction errors, where false positives and false negatives carry asymmetric consequences.

Validation was performed using a strict train/validation split, with all preprocessing steps fitted only on the training portion to avoid leakage.

---

## Hyperparameter Optimization

Hyperparameter tuning was conducted using Randomized Search to balance exploration of the parameter space with computational efficiency. The objective was to identify robust configurations rather than over-specialized solutions.

Importantly, hyperparameter optimization was kept outside the final production pipeline. Only the selected, fully trained model and its associated decision threshold are passed downstream.

---

## Production Pipeline

The final artifact of this project is a fully encapsulated scikit-learn Pipeline that includes preprocessing, feature engineering, and the trained model.

By consolidating all logic into a single pipeline, the project avoids training–serving skew and ensures identical behavior between offline evaluation and live inference. This design also simplifies deployment into batch scoring or real-time systems.

Threshold selection is treated as an explicit modeling decision and fixed prior to deployment.

---

## Results and Interpretation

The final model demonstrates strong and stable performance on validation data, while maintaining interpretability and operational simplicity.

More importantly than the absolute metrics, the resulting system reflects a set of engineering principles: disciplined validation, reproducibility, and alignment between modeling choices and business objectives.

---

## Tooling

The project was implemented using Python, with Pandas and NumPy for data manipulation, scikit-learn for modeling and pipeline construction, and Matplotlib / Seaborn for targeted visualization. Development and experimentation were conducted in Jupyter Notebook.

---

## Future Work

Potential extensions include model monitoring and drift detection, cost-sensitive optimization, and deployment behind a REST API for real-time inference.
