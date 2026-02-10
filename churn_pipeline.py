import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import FixedThresholdClassifier


# Custom Transformers

class ColumnNameCleaner(BaseEstimator, TransformerMixin):
    """Standardize column names: lowercase, replace spaces with underscores."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X.columns = X.columns.str.strip().str.lower().str.replace(' ', '_')
        return X

class NumericConverter(BaseEstimator, TransformerMixin):
    """Convert totalcharges to numeric and handle whitespace/missing values."""
    def __init__(self):
        self.fill_value_ = None
    
    def fit(self, X, y=None):
        X = X.copy()
        # Ensure column names are standardized if this is running independently
        if 'TotalCharges' in X.columns:
            col = 'TotalCharges'
        else:
            col = 'totalcharges'
            
        # Temporarily convert to find median
        temp_col = pd.to_numeric(X[col], errors='coerce')
        self.fill_value_ = temp_col.median()
        return self
    
    def transform(self, X):
        X = X.copy()
        # Handle capitalization variance if preceding steps failed
        col = 'totalcharges' if 'totalcharges' in X.columns else 'TotalCharges'
        
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(self.fill_value_)
        return X

class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Standardize categorical string values: lowercase, replace spaces/dashes."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].str.replace(' ', '_').str.replace('-', '_').str.lower()
        return X

class TotalServicesCreator(BaseEstimator, TransformerMixin):
    """Create total_services feature by counting active add-on services."""
    def __init__(self):
        self.service_columns = [
            'onlinesecurity', 'onlinebackup', 'deviceprotection', 
            'techsupport', 'streamingtv', 'streamingmovies'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Only use columns that actually exist in X
        available_cols = [col for col in self.service_columns if col in X.columns]
        
        # Count 'yes' in these columns
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
        if 'total_services' in X.columns:
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
        
        # Gender: Female=1, Male=0
        if 'gender' in X.columns:
            X['gender'] = X['gender'].map({'female': 1, 'male': 0})
            
        # Binary Yes/No columns
        for col in self.binary_cols:
            if col in X.columns and col != 'gender':
                X[col] = X[col].map({'yes': 1, 'no': 0})
        
        # Service columns (Yes/No/No internet service)
        for col in self.service_cols:
            if col in X.columns:
                X[col] = X[col].map({'yes': 1, 'no': 0, 'no_phone_service': 0, 'no_internet_service': 0})
        
        # Target variable if present
        if self.target_col in X.columns:
            X[self.target_col] = X[self.target_col].map({'yes': 1, 'no': 0})
        
        return X

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode specified nominal categorical columns."""
    def __init__(self):
        self.categorical_cols = ['internetservice', 'contract', 'paymentmethod']
        self.encoded_columns_ = None
    
    def fit(self, X, y=None):
        # Determine the columns that will be generated
        X_temp = X.copy()
        available_cols = [col for col in self.categorical_cols if col in X_temp.columns]
        
        if available_cols:
            dummies = pd.get_dummies(X_temp[available_cols], prefix=available_cols, drop_first=True)
            self.encoded_columns_ = dummies.columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        available_cols = [col for col in self.categorical_cols if col in X.columns]
        
        if available_cols:
            # Generate dummies for current data
            dummies = pd.get_dummies(X[available_cols], prefix=available_cols, drop_first=True)
            
            # Ensure all columns from fit exist (handle missing categories in test data)
            if self.encoded_columns_ is not None:
                for col in self.encoded_columns_:
                    if col not in dummies.columns:
                        dummies[col] = 0
                # Reorder to match fit structure
                dummies = dummies[self.encoded_columns_]
            
            X = X.drop(columns=available_cols)
            X = pd.concat([X, dummies], axis=1)
        
        return X


# Main Wrapper Class

class ChurnPredictor:
    """
    Wrapper class to load the pipeline and make predictions.
    """
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)