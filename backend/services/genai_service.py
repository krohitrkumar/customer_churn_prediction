import os
import json
from groq import Groq
from database.config import settings

class GenAIService:
    def __init__(self):
        self.client = None
        self.model = settings.GROQ_MODEL or "qwen/qwen3.8-27b"
        self._initialize_client()

    def _initialize_client(self):
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if api_key and len(api_key.strip()) > 5:
            try:
                self.client = Groq(api_key=api_key.strip())
            except Exception:
                self.client = None
        else:
            self.client = None

    def generate_retention_copilot(
        self,
        customer_profile: dict,
        churn_probability: float,
        shap_explanation: dict,
        manager_name: str = "Rohit Kumar",
        manager_role: str = "Customer Success Lead, Retentrix"
    ) -> dict:
        if self.client is None:
            self._initialize_client()

        risk_drivers = [item["feature"] for item in shap_explanation.get("top_risk_drivers", [])]
        protective = [item["feature"] for item in shap_explanation.get("top_protective_factors", [])]

        cust_first = customer_profile.get("first_name", "Valued")
        cust_last = customer_profile.get("last_name", "Customer")
        full_name = f"{cust_first} {cust_last}".strip()
        tenure = customer_profile.get("tenure_months", 12)
        satisfaction = customer_profile.get("satisfaction_score", 5.0)
        calls = customer_profile.get("support_calls", 0)
        late = customer_profile.get("late_payments", 0)
        contract = customer_profile.get("contract_type", "month_to_month")
        payment = customer_profile.get("payment_method", "card")

        system_prompt = (
            "You are Retentrix Copilot, the AI retention intelligence engine for Retentrix. "
            "Analyze the customer metrics and SHAP factors. "
            "Respond ONLY with a single valid JSON object containing these exact keys: "
            "'root_cause_bullets' (list of 2-3 concise strings), "
            "'strategy_steps' (list of 2-3 actionable strings), "
            "'recommended_incentive' (string), "
            "'urgency' (string: 'Critical', 'High', or 'Moderate'), "
            "'economic_viability' (string: explain whether customer is high ROI to save or a high-drain account at risk of negative ROI), "
            "'email_subject' (string), "
            "'email_body' (string). "
            "In email_body: address the customer warmly by their full name, explicitly reference their exact tenure and satisfaction score, "
            "offer the incentive, and sign off as: "
            f"Best regards,\n{manager_name}\n{manager_role}"
        )

        user_content = (
            f"Customer Full Name: {full_name}\n"
            f"Tenure with us: {tenure} months\n"
            f"Satisfaction Score: {satisfaction}/10.0\n"
            f"Support Calls: {calls}\n"
            f"Late Payments: {late}\n"
            f"Contract Type: {contract}\n"
            f"Payment Method: {payment}\n"
            f"Predicted Churn Risk: {churn_probability}%\n"
            f"Primary SHAP Risk Drivers: {', '.join(risk_drivers) if risk_drivers else 'None'}\n"
            f"Primary SHAP Protective Factors: {', '.join(protective) if protective else 'None'}\n"
            f"Account Manager: {manager_name}\n"
        )

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=1200
                )
                raw_text = response.choices[0].message.content.strip()
                data = json.loads(raw_text)
                return data
            except Exception:
                pass

        urgency_level = "Critical" if churn_probability > 75.0 else ("High" if churn_probability > 40.0 else "Moderate")
        viability = "Caution: High Drain Account - High support volume and low satisfaction indicate low recovery probability" if calls > 10 or satisfaction < 3.0 else "High ROI - Account is economically viable to retain with targeted intervention"

        fallback_bullets = [
            f"Churn probability is {churn_probability}%, driven primarily by {', '.join(risk_drivers[:2]) if risk_drivers else 'recent engagement changes'}.",
            f"Customer has recorded {calls} support inquiries and satisfaction rating sits at {satisfaction}/10.0.",
            f"Contract commitment ({contract}) requires proactive renewal incentive."
        ]

        fallback_steps = [
            f"Schedule a 1-on-1 account health review with {manager_name} within 48 hours.",
            "Review outstanding support tickets and ensure all technical blockers are permanently resolved.",
            "Propose a targeted contract renewal incentive with locked-in pricing."
        ]

        fallback_incentive = "15% discount on 12-month agreement extension + 2 months complimentary priority SLA support"

        fallback_email = (
            f"Hi {full_name},\n\n"
            f"Thank you for being a valued Retentrix partner for the past {tenure} months. "
            f"I was recently reviewing your account activity and noticed your recent satisfaction score of {satisfaction}/10. "
            f"Ensuring your team receives exceptional service is my top priority.\n\n"
            f"I would like to personally offer your team an exclusive {fallback_incentive} to make sure you have everything needed to succeed.\n\n"
            f"Could we connect for a brief 10-minute check-in call this week?\n\n"
            f"Best regards,\n{manager_name}\n{manager_role}"
        )

        return {
            "root_cause_bullets": fallback_bullets,
            "strategy_steps": fallback_steps,
            "recommended_incentive": fallback_incentive,
            "urgency": urgency_level,
            "economic_viability": viability,
            "email_subject": f"Enhancing your Retentrix experience, {cust_first}",
            "email_body": fallback_email
        }

genai_service = GenAIService()
