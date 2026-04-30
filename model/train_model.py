import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from utils.feature_engineering import FeatureEngineer

def train_fraud_model(data_path):
    """Train XGBoost model for fraud detection"""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution: {df['is_fraud'].value_counts()}")
    
    # Prepare features
    feature_cols = ['transaction_id', 'user_id', 'amount', 'transaction_type', 
                   'merchant_category', 'country', 'hour', 'device_risk_score', 
                   'ip_risk_score', 'is_fraud']
    
    df_model = df[feature_cols].copy()
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Prepare features and target
    X = df_model.drop(['is_fraud', 'transaction_id'], axis=1)
    y = df_model['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature engineering
    print("Engineering features...")
    fe.fit(X_train)
    
    X_train_processed, feature_names = fe.transform(X_train)
    X_test_processed, _ = fe.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_processed.shape}")
    
    # Handle class imbalance with scale_pos_weight
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    # XGBoost model
    print("Training XGBoost model...")
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Train model
    xgb_model.fit(
        X_train_processed, y_train,
        eval_set=[(X_test_processed, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred = xgb_model.predict(X_test_processed)
    y_pred_proba = xgb_model.predict_proba(X_test_processed)[:, 1]
    
    # Evaluation
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 10 Important Features ===")
    print(feature_importance.head(10))
    
    # Save model and feature engineer
    print("\nSaving model and preprocessing pipeline...")
    joblib.dump(xgb_model, 'model/xgboost_model.pkl')
    joblib.dump(fe, 'model/feature_engineer.pkl')
    joblib.dump(feature_names, 'model/feature_names.pkl')
    
    # Create visualizations
    create_evaluation_plots(y_test, y_pred, y_pred_proba, feature_importance)
    
    return xgb_model, fe, feature_names

def create_evaluation_plots(y_test, y_pred, y_pred_proba, feature_importance):
    """Create evaluation visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    
    # Feature Importance
    top_features = feature_importance.head(10)
    axes[1, 0].barh(range(len(top_features)), top_features['importance'])
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels(top_features['feature'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top 10 Feature Importance')
    
    # Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Non-Fraud', color='green')
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Fraud', color='red')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model/evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Train the model
    model, fe, feature_names = train_fraud_model('data/synthetic_fraud_dataset.csv')
    print("\nModel training completed successfully!")
