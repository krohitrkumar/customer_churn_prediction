"""
ML Pipeline Training Script
Extracts data, runs preprocessing, trains GradientBoostingClassifier, and serializes artifact.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "churn_model.pkl")

def train_and_save_model():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    cat_cols = ['region', 'contract_type', 'payment_method']
    num_cols = ['tenure_months', 'support_calls', 'late_payments', 'satisfaction_score']

    X = df[['late_payments', 'support_calls', 'satisfaction_score', 'tenure_months', 'contract_type', 'payment_method', 'region']]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline([
        ('Preprocessing', preprocessor),
        ('Model', GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=2,
            min_samples_split=5,
            n_estimators=200,
            random_state=42
        ))
    ])

    print("Training Gradient Boosting Pipeline...")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n--- Model Evaluation ---")
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Test Recall   : {rec * 100:.2f}%")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"\nModel saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
