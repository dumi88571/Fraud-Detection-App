import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def train_fraud_model(data_path):
    """Train Random Forest model for fraud detection (more compatible)"""
    
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
    
    # Feature engineering
    print("Engineering features...")
    
    # Create time-based features
    df_model['hour_sin'] = np.sin(2 * np.pi * df_model['hour'] / 24)
    df_model['hour_cos'] = np.cos(2 * np.pi * df_model['hour'] / 24)
    df_model['is_weekend'] = (df_model['hour'] >= 0) & (df_model['hour'] <= 6) | (df_model['hour'] >= 22)
    
    # Create risk score combinations
    df_model['combined_risk_score'] = (df_model['device_risk_score'] + df_model['ip_risk_score']) / 2
    df_model['risk_diff'] = abs(df_model['device_risk_score'] - df_model['ip_risk_score'])
    
    # Amount-based features
    df_model['amount_log'] = np.log1p(df_model['amount'])
    df_model['is_high_amount'] = df_model['amount'] > df_model['amount'].quantile(0.95)
    
    # Encode categorical variables
    le_transaction = LabelEncoder()
    le_merchant = LabelEncoder()
    le_country = LabelEncoder()
    
    df_model['transaction_type_encoded'] = le_transaction.fit_transform(df_model['transaction_type'])
    df_model['merchant_category_encoded'] = le_merchant.fit_transform(df_model['merchant_category'])
    df_model['country_encoded'] = le_country.fit_transform(df_model['country'])
    
    # Select final features
    feature_columns = [
        'amount', 'hour', 'device_risk_score', 'ip_risk_score',
        'hour_sin', 'hour_cos', 'is_weekend', 'combined_risk_score',
        'risk_diff', 'amount_log', 'is_high_amount',
        'transaction_type_encoded', 'merchant_category_encoded', 'country_encoded'
    ]
    
    # Prepare features and target
    X = df_model[feature_columns]
    y = df_model['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_scaled.shape}")
    
    # Random Forest model
    print("Training Random Forest model...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    # Train model
    rf_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluation
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 10 Important Features ===")
    print(feature_importance.head(10))
    
    # Save model and preprocessing components
    print("\nSaving model and preprocessing pipeline...")
    joblib.dump(rf_model, 'model/rf_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(feature_columns, 'model/feature_columns.pkl')
    
    # Save encoders
    joblib.dump(le_transaction, 'model/le_transaction.pkl')
    joblib.dump(le_merchant, 'model/le_merchant.pkl')
    joblib.dump(le_country, 'model/le_country.pkl')
    
    # Create visualizations
    create_evaluation_plots(y_test, y_pred, y_pred_proba, feature_importance)
    
    return rf_model, scaler, feature_columns, feature_importance

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
    model, scaler, feature_names, feature_importance = train_fraud_model('data/synthetic_fraud_dataset.csv')
    print("\nModel training completed successfully!")
