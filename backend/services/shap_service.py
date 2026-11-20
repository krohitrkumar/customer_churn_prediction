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

        contract_label = contract.replace("_", " ").title()
        payment_label = payment.replace("_", " ").title()
        region_label = region.replace("_", " ").title()

        grouped_impacts = {}
        for name, impact in zip(feature_names, values):
            val = float(impact)
            if name.startswith("cat__contract_type_"):
                grouped_impacts[f"Contract: {contract_label}"] = grouped_impacts.get(f"Contract: {contract_label}", 0.0) + val
            elif name.startswith("cat__payment_method_"):
                grouped_impacts[f"Payment: {payment_label}"] = grouped_impacts.get(f"Payment: {payment_label}", 0.0) + val
            elif name.startswith("cat__region_"):
                grouped_impacts[f"Region: {region_label}"] = grouped_impacts.get(f"Region: {region_label}", 0.0) + val
            elif "late_payments" in name:
                grouped_impacts["Late Payments"] = val
            elif "support_calls" in name:
                grouped_impacts["Support Calls"] = val
            elif "satisfaction_score" in name:
                grouped_impacts["Satisfaction Score"] = val
            elif "tenure_months" in name:
                grouped_impacts["Customer Tenure"] = val
            else:
                grouped_impacts[name] = val

        factors = []
        for feat_name, total_impact in grouped_impacts.items():
            if abs(total_impact) >= 0.02:
                factors.append({
                    "feature": feat_name,
                    "impact": round(total_impact, 4),
                    "is_risk_driver": total_impact > 0
                })

        factors.sort(key=lambda x: abs(x["impact"]), reverse=True)

        risk_drivers = [f for f in factors if f["impact"] >= 0.02][:4]
        protective_factors = [f for f in factors if f["impact"] <= -0.02][:4]

        base_value = float(self.explainer.expected_value[0]) if hasattr(self.explainer.expected_value, "__iter__") else float(self.explainer.expected_value)

        return {
            "base_value": round(base_value, 4),
            "all_factors": factors,
            "top_risk_drivers": risk_drivers,
            "top_protective_factors": protective_factors
        }

shap_service = SHAPService()
