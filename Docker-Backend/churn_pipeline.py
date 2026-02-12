"""
Churn Prediction Pipeline - Production Module

This module contains all custom transformers and helper classes needed to load
and use the saved churn prediction pipeline. All transformer classes match exactly
with those defined in the churn_prediction.ipynb notebook.

Usage:
    from churn_pipeline import ChurnPredictor
    
    predictor = ChurnPredictor('churn_prediction_pipeline.pkl')
    predictions = predictor.predict(new_data)
    probabilities = predictor.predict_proba(new_data)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# =============================================================================
# Custom Transformers (matching notebook definitions exactly)
# =============================================================================

class ColumnNameCleaner(BaseEstimator, TransformerMixin):
    """Standardize column names: lowercase, replace spaces with underscores."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X.columns = X.columns.str.strip().str.lower().str.replace(' ', '_')
        return X


class NumericConverter(BaseEstimator, TransformerMixin):
    """Convert totalcharges to numeric and handle whitespace values."""
    
    def __init__(self):
        self.fill_value_ = None
    
    def fit(self, X, y=None):
        X = X.copy()
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')
        self.fill_value_ = X['totalcharges'].median()
        return self
    
    def transform(self, X):
        X = X.copy()
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')
        X['totalcharges'] = X['totalcharges'].fillna(self.fill_value_)
        return X


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Standardize categorical values: lowercase, replace spaces/dashes with underscores."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].str.replace(' ', '_').str.replace('-', '_').str.lower()
        return X


class TotalServicesCreator(BaseEstimator, TransformerMixin):
    """Create total_services feature by counting active services."""
    
    def __init__(self):
        self.service_columns = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 
                                'techsupport', 'streamingtv', 'streamingmovies']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        available_cols = [col for col in self.service_columns if col in X.columns]
        X['total_services'] = (X[available_cols] == 'yes').sum(axis=1)
        return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drop specified columns from dataframe."""
    
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        existing_cols = [col for col in self.columns_to_drop if col in X.columns]
        return X.drop(columns=existing_cols)


class LowServicesFeature(BaseEstimator, TransformerMixin):
    """Create binary feature for customers with low service counts."""
    
    def __init__(self, threshold=1):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["low_services"] = (X["total_services"] <= self.threshold).astype(int)
        return X


class CustomEncoder(BaseEstimator, TransformerMixin):
    """Encode binary categorical variables with custom mappings (Yes/No -> 1/0)."""
    
    def __init__(self):
        self.binary_cols = ['gender', 'partner', 'dependents', 'phoneservice', 'paperlessbilling']
        self.service_cols = ['multiplelines', 'onlinesecurity', 'onlinebackup', 
                            'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']
        self.target_col = 'churn'
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.binary_cols:
            if col in X.columns:
                X[col] = X[col].map({'yes': 1, 'no': 0, 'male': 0, 'female': 1})
        
        for col in self.service_cols:
            if col in X.columns:
                X[col] = X[col].map({'yes': 1, 'no': 0, 'no_phone_service': 0, 'no_internet_service': 0})
        
        if self.target_col in X.columns:
            X[self.target_col] = X[self.target_col].map({'yes': 1, 'no': 0})
        
        return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode specified nominal categorical columns."""
    
    def __init__(self):
        self.categorical_cols = ['internetservice', 'contract', 'paymentmethod']
        self.encoded_columns_ = None
    
    def fit(self, X, y=None):
        X = X.copy()
        available_cols = [col for col in self.categorical_cols if col in X.columns]
        
        if available_cols:
            dummies = pd.get_dummies(X[available_cols], prefix=available_cols, drop_first=True)
            self.encoded_columns_ = dummies.columns.tolist()
        
        return self
    
    def transform(self, X):
        X = X.copy()
        available_cols = [col for col in self.categorical_cols if col in X.columns]
        
        if available_cols:
            dummies = pd.get_dummies(X[available_cols], prefix=available_cols, drop_first=True)
            
            for col in self.encoded_columns_:
                if col not in dummies.columns:
                    dummies[col] = 0
            
            dummies = dummies[self.encoded_columns_]
            
            X = X.drop(columns=available_cols)
            X = pd.concat([X, dummies], axis=1)
        
        return X


# =============================================================================
# Main Wrapper Class
# =============================================================================

class ChurnPredictor:
    """
    Wrapper class to load and use the churn prediction pipeline.
    
    This class provides a simple interface for making predictions with the
    saved pipeline model.
    
    Parameters
    ----------
    model_path : str
        Path to the saved pipeline pickle file
        
    Attributes
    ----------
    model : Pipeline
        The loaded scikit-learn pipeline
        
    Examples
    --------
    >>> predictor = ChurnPredictor('churn_prediction_pipeline.pkl')
    >>> predictions = predictor.predict(new_customer_data)
    >>> probabilities = predictor.predict_proba(new_customer_data)
    """
    
    def __init__(self, model_path):
        """Load the trained pipeline from disk."""
        self.model = joblib.load(model_path)
        print(f"✓ Pipeline loaded successfully from {model_path}")

    def predict(self, X):
        """
        Predict churn labels for input data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input customer data with original column names
            
        Returns
        -------
        np.ndarray
            Binary predictions (0 = No Churn, 1 = Churn)
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict churn probabilities for input data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input customer data with original column names
            
        Returns
        -------
        np.ndarray
            Probability estimates for each class [P(No Churn), P(Churn)]
        """
        return self.model.predict_proba(X)
    
    def get_feature_names(self):
        """
        Get the names of features used by the model.
        
        Returns
        -------
        list
            List of feature names after preprocessing
        """
        # Get the preprocessing pipeline
        preprocessing = self.model.named_steps['preprocessing']
        
        # Create a dummy dataframe to get feature names
        # This assumes you have access to sample data
        print("Note: To get exact feature names, pass a sample dataframe through preprocessing")
        return None
