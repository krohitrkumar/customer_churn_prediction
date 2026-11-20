import os
import joblib
import pandas as pd
import shap
from database.config import settings

class SHAPService:
    def __init__(self):
        self.pipeline = None
        self.preprocessor = None
        self.model = None
        self.explainer = None
        self._initialize()

    def _initialize(self):
        if not os.path.exists(settings.MODEL_PATH):
            return

        try:
            self.pipeline = joblib.load(settings.MODEL_PATH)
            self.preprocessor = self.pipeline.named_steps["Preprocessing"]
            self.model = self.pipeline.named_steps["Model"]
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.pipeline = None
            self.preprocessor = None
            self.model = None
            self.explainer = None

    def explain_customer(self, customer_data: dict) -> dict:
        if self.explainer is None:
            self._initialize()
            if self.explainer is None:
                return {"drivers": [], "protective": [], "base_value": 0.0}

        contract = customer_data["contract_type"].value if hasattr(customer_data["contract_type"], "value") else str(customer_data["contract_type"])
        payment = customer_data["payment_method"].value if hasattr(customer_data["payment_method"], "value") else str(customer_data["payment_method"])
        region = customer_data["region"].value if hasattr(customer_data["region"], "value") else str(customer_data["region"])

        df = pd.DataFrame([{
            "tenure_months": customer_data.get("tenure_months", 12),
            "support_calls": customer_data.get("support_calls", 0),
            "late_payments": customer_data.get("late_payments", 0),
            "satisfaction_score": customer_data.get("satisfaction_score", 5.0),
            "region": region,
            "contract_type": contract,
            "payment_method": payment
        }])

        transformed_features = self.preprocessor.transform(df)
        feature_names = self.preprocessor.get_feature_names_out()
        shap_values = self.explainer.shap_values(transformed_features)

        if isinstance(shap_values, list):
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 2:
            values = shap_values[0]
        else:
            values = shap_values

        readable_name_map = {
            "remainder__late_payments": "Late Payments",
            "remainder__support_calls": "Support Calls",
            "remainder__satisfaction_score": "Satisfaction Score",
            "remainder__tenure_months": "Customer Tenure",
            "cat__contract_type_one_year": "One-Year Contract",
            "cat__contract_type_two_year": "Two-Year Contract",
            "cat__payment_method_card": "Card Payment Method",
            "cat__payment_method_wallet": "Wallet Payment Method",
            "cat__region_asia": "Asia Region",
            "cat__region_europe": "Europe Region",
            "cat__region_latam": "LatAm Region",
            "cat__region_north_america": "North America Region",
            "cat__region_south_america": "South America Region"
        }

        factors = []
        for name, impact in zip(feature_names, values):
            clean_name = readable_name_map.get(name, name.replace("remainder__", "").replace("cat__", "").replace("_", " ").title())
            factors.append({
                "feature": clean_name,
                "impact": round(float(impact), 4),
                "is_risk_driver": float(impact) > 0
            })

        factors.sort(key=lambda x: abs(x["impact"]), reverse=True)

        risk_drivers = [f for f in factors if f["impact"] > 0][:4]
        protective_factors = [f for f in factors if f["impact"] < 0][:4]

        base_value = float(self.explainer.expected_value[0]) if hasattr(self.explainer.expected_value, "__iter__") else float(self.explainer.expected_value)

        return {
            "base_value": round(base_value, 4),
            "all_factors": factors,
            "top_risk_drivers": risk_drivers,
            "top_protective_factors": protective_factors
        }

shap_service = SHAPService()
