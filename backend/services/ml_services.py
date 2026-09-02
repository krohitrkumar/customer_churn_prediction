import os   
import joblib
import pandas as pd
from database.config import settings
from schemas.prediction import PlaybookRecommendation

class MLService: 
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(settings.MODEL_PATH):
            try:
                self.model = joblib.load(settings.MODEL_PATH)
                print(f" ML Model loaded successfully from {settings.MODEL_PATH}")
            except Exception as e:
                print(f" Failed to load model: {e}")
                self.model = None
        else: 
            print(f"Model path not found: {settings.MODEL_PATH}")
            self.model = None

    def predict_churn(self, data: dict):
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise RuntimeError("ML Model artifact is missing or failed to load.")

        # Safely convert Enums to strings for Scikit-Learn
        contract_type = data["contract_type"].value if hasattr(data["contract_type"], "value") else str(data["contract_type"])
        payment_method = data["payment_method"].value if hasattr(data["payment_method"], "value") else str(data["payment_method"])
        region = data["region"].value if hasattr(data["region"], "value") else str(data["region"])

        # Structure DataFrame with exact columns expected by the trained pipeline
        df = pd.DataFrame([{
            "late_payments": data["late_payments"],
            "support_calls": data["support_calls"],
            "satisfaction_score": data["satisfaction_score"],
            "tenure_months": data["tenure_months"],
            "contract_type": contract_type,
            "payment_method": payment_method,
            "region": region
        }])


        prediction = int(self.model.predict(df)[0])
        prob = float(self.model.predict_proba(df)[0][1])
        prob_pct = round(prob * 100, 2)

        if prob > 0.75:
            risk = "Critical"
        elif prob > 0.40:
            risk = "Moderate"
        else:
            risk = "Low"

        playbooks = []
        if data["satisfaction_score"] <= 3.0:
            playbooks.append(PlaybookRecommendation(
                icon="alert-circle",
                category="Customer Satisfaction",
                action="Trigger immediate executive outreach and dispatch a customer feedback survey."
            ))
        if data["support_calls"] >= 5:
            playbooks.append(PlaybookRecommendation(
                icon="phone-call",
                category="Support Queue Priority",
                action="Assign a Senior Technical Specialist to resolve outstanding friction tickets."
            ))
        if data["late_payments"] >= 3:
            playbooks.append(PlaybookRecommendation(
                icon="credit-card",
                category="Billing Flexibility",
                action="Offer payment restructuring or automated installment reminder workflow."
            ))
        if data["tenure_months"] < 12:
            playbooks.append(PlaybookRecommendation(
                icon="clock",
                category="Onboarding Retention",
                action="Enroll customer in high-touch onboarding check-in calls."
            ))
        if contract_type in ["month_to_month"]:
            playbooks.append(PlaybookRecommendation(
                icon="file-text",
                category="Contract Commitment",
                action="Offer a 15% discount incentive for upgrading to an annual subscription plan."
            ))
        if not playbooks:
            playbooks.append(PlaybookRecommendation(
                icon="check-circle",
                category="Healthy Account",
                action="Account is stable. Maintain standard automated quarterly product updates."
            ))


        return {
            "churn_prediction": prediction,
            "churn_probability": prob_pct,
            "risk_level": risk,
            "playbooks": playbooks
        }

ml_service = MLService()