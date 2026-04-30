import pandas as pd
import numpy as np

class MockFraudModel:
    def __init__(self):
        self.feature_columns = [
            'amount', 'hour', 'device_risk_score', 'ip_risk_score',
            'transaction_type', 'merchant_category', 'country'
        ]
        
    def predict(self, X):
        """Simple rule-based prediction"""
        if isinstance(X, pd.DataFrame):
            # Simple fraud detection rules
            risk_score = (
                (X['amount'] > 1000).astype(int) * 0.3 +
                X['device_risk_score'] * 0.3 +
                X['ip_risk_score'] * 0.3 +
                ((X['hour'] < 6) | (X['hour'] > 22)).astype(int) * 0.1
            )
            return (risk_score > 0.5).astype(int)
        return np.array([0] * len(X))
    
    def predict_proba(self, X):
        """Return probability scores"""
        if isinstance(X, pd.DataFrame):
            # Simple fraud detection rules
            risk_score = (
                (X['amount'] > 1000).astype(int) * 0.3 +
                X['device_risk_score'] * 0.3 +
                X['ip_risk_score'] * 0.3 +
                ((X['hour'] < 6) | (X['hour'] > 22)).astype(int) * 0.1
            )
            prob_fraud = np.clip(risk_score, 0, 1)
            prob_legit = 1 - prob_fraud
            return np.column_stack([prob_legit, prob_fraud])
        return np.array([[0.9, 0.1]] * len(X))
