import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        
    def fit(self, X):
        """Fit the feature engineering pipeline on training data"""
        X_processed = X.copy()
        
        # Create time-based features
        X_processed['hour_sin'] = np.sin(2 * np.pi * X_processed['hour'] / 24)
        X_processed['hour_cos'] = np.cos(2 * np.pi * X_processed['hour'] / 24)
        X_processed['is_weekend'] = (X_processed['hour'] >= 0) & (X_processed['hour'] <= 6) | (X_processed['hour'] >= 22)
        
        # Create risk score combinations
        X_processed['combined_risk_score'] = (X_processed['device_risk_score'] + X_processed['ip_risk_score']) / 2
        X_processed['risk_diff'] = abs(X_processed['device_risk_score'] - X_processed['ip_risk_score'])
        
        # Amount-based features
        X_processed['amount_log'] = np.log1p(X_processed['amount'])
        X_processed['is_high_amount'] = X_processed['amount'] > X_processed['amount'].quantile(0.95)
        
        # Encode categorical variables
        categorical_cols = ['transaction_type', 'merchant_category', 'country']
        for col in categorical_cols:
            if col in X_processed.columns:
                le = LabelEncoder()
                X_processed[f'{col}_encoded'] = le.fit_transform(X_processed[col])
                self.label_encoders[col] = le
        
        # Select final features
        self.feature_columns = [
            'amount', 'hour', 'device_risk_score', 'ip_risk_score',
            'hour_sin', 'hour_cos', 'is_weekend', 'combined_risk_score',
            'risk_diff', 'amount_log', 'is_high_amount',
            'transaction_type_encoded', 'merchant_category_encoded', 'country_encoded'
        ]
        
        # Filter existing columns
        self.feature_columns = [col for col in self.feature_columns if col in X_processed.columns]
        
        # Fit scaler
        self.scaler.fit(X_processed[self.feature_columns])
        self.is_fitted = True
        
        return self
    
    def transform(self, X):
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transforming data")
        
        X_processed = X.copy()
        
        # Create time-based features
        X_processed['hour_sin'] = np.sin(2 * np.pi * X_processed['hour'] / 24)
        X_processed['hour_cos'] = np.cos(2 * np.pi * X_processed['hour'] / 24)
        X_processed['is_weekend'] = (X_processed['hour'] >= 0) & (X_processed['hour'] <= 6) | (X_processed['hour'] >= 22)
        
        # Create risk score combinations
        X_processed['combined_risk_score'] = (X_processed['device_risk_score'] + X_processed['ip_risk_score']) / 2
        X_processed['risk_diff'] = abs(X_processed['device_risk_score'] - X_processed['ip_risk_score'])
        
        # Amount-based features
        X_processed['amount_log'] = np.log1p(X_processed['amount'])
        X_processed['is_high_amount'] = X_processed['amount'] > 1000  # Default threshold
        
        # Encode categorical variables
        categorical_cols = ['transaction_type', 'merchant_category', 'country']
        for col in categorical_cols:
            if col in X_processed.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                unique_values = set(X_processed[col].unique())
                known_values = set(le.classes_)
                unknown_values = unique_values - known_values
                
                if unknown_values:
                    # Map unknown values to most common class
                    most_common = le.classes_[0]
                    X_processed[col] = X_processed[col].apply(lambda x: most_common if x in unknown_values else x)
                
                X_processed[f'{col}_encoded'] = le.transform(X_processed[col])
        
        # Filter existing columns
        available_features = [col for col in self.feature_columns if col in X_processed.columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed[available_features])
        
        return X_scaled, available_features
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def get_feature_importance_names(self):
        """Get feature names for importance plotting"""
        return self.feature_columns if self.feature_columns else []
