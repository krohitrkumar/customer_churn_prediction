import numpy as np

class ContextualBanditService:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.feature_dim = 6
        self.arms = [
            {
                "id": 0,
                "name": "Standard Automated Nudge",
                "category": "Digital Touchpoint",
                "cost_usd": 0.50,
                "description": "Send automated personalized product usage digest and tips."
            },
            {
                "id": 1,
                "name": "CSM Concierge Outreach",
                "category": "Customer Success",
                "cost_usd": 10.00,
                "description": "Assign dedicated Customer Success Specialist for a priority check-in."
            },
            {
                "id": 2,
                "name": "15% Renewal Loyalty Discount",
                "category": "Financial Incentive",
                "cost_usd": 20.00,
                "description": "Offer contract commitment discount for upgrading to annual billing."
            },
            {
                "id": 3,
                "name": "Free Enterprise Feature Tier Upgrade",
                "category": "Product Expansion",
                "cost_usd": 40.00,
                "description": "Grant 60-day complimentary access to premium analytics and priority SLA."
            },
            {
                "id": 4,
                "name": "Executive Sponsor & Custom Terms Review",
                "category": "VIP Escalation",
                "cost_usd": 75.00,
                "description": "Executive leadership outreach with bespoke SLAs and custom contract restructuring."
            }
        ]
        self.A = [np.identity(self.feature_dim) for _ in range(len(self.arms))]
        self.b = [np.zeros((self.feature_dim, 1)) for _ in range(len(self.arms))]
        self._warm_start_policy()

    def _warm_start_policy(self):
        sample_profiles = [
            (np.array([[1.0, 0.8, 0.05, 0.0, 0.9, 0.05]]).T, 0, 18.0),
            (np.array([[1.0, 0.6, 0.1, 0.0, 0.8, 0.15]]).T, 0, 22.0),
            (np.array([[1.0, 0.5, 0.25, 0.1, 0.6, 0.35]]).T, 1, 35.0),
            (np.array([[1.0, 0.3, 0.4, 0.2, 0.45, 0.60]]).T, 2, 55.0),
            (np.array([[1.0, 0.2, 0.6, 0.3, 0.3, 0.75]]).T, 3, 78.0),
            (np.array([[1.0, 0.1, 0.85, 0.5, 0.15, 0.95]]).T, 4, 125.0)
        ]
        for x, arm_id, reward in sample_profiles:
            self.A[arm_id] += np.dot(x, x.T) * 2.0
            self.b[arm_id] += reward * x * 2.0

    def _extract_context(self, customer_data: dict, churn_probability: float) -> np.ndarray:
        tenure = min(72.0, max(1.0, float(customer_data.get("tenure_months", 12)))) / 72.0
        calls = min(20.0, max(0.0, float(customer_data.get("support_calls", 0)))) / 20.0
        late = min(12.0, max(0.0, float(customer_data.get("late_payments", 0)))) / 12.0
        sat = min(10.0, max(1.0, float(customer_data.get("satisfaction_score", 5.0)))) / 10.0
        prob = min(100.0, max(0.0, float(churn_probability))) / 100.0

        return np.array([[1.0, tenure, calls, late, sat, prob]]).T

    def select_action(self, customer_data: dict, churn_probability: float) -> dict:
        x = self._extract_context(customer_data, churn_probability)
        arm_evaluations = []
        prob = float(churn_probability) / 100.0
        ltv_estimate = 300.0 + (float(customer_data.get("tenure_months", 12)) * 18.0)

        uplift_rates = [0.10, 0.25, 0.45, 0.65, 0.85]

        for i, arm in enumerate(self.arms):
            A_inv = np.linalg.pinv(self.A[i])
            theta = np.dot(A_inv, self.b[i])
            learned_score = float(np.dot(theta.T, x)[0][0])
            variance = float(np.dot(np.dot(x.T, A_inv), x)[0][0])
            exploration_bonus = self.alpha * np.sqrt(max(0.0, variance))

            expected_saved_revenue = ltv_estimate * prob * uplift_rates[i]
            net_roi = round(expected_saved_revenue - arm["cost_usd"], 2)

            total_utility = (net_roi * 0.7) + (learned_score * 0.3) + exploration_bonus

            arm_evaluations.append({
                "arm_id": arm["id"],
                "name": arm["name"],
                "category": arm["category"],
                "cost_usd": arm["cost_usd"],
                "description": arm["description"],
                "net_roi_usd": net_roi,
                "ucb_score": round(total_utility, 2),
                "expected_saved_revenue": round(expected_saved_revenue, 2),
                "exploration_bonus": round(exploration_bonus, 2)
            })

        best_arm = max(arm_evaluations, key=lambda a: a["ucb_score"])

        return {
            "selected_action": best_arm,
            "all_actions": arm_evaluations,
            "algorithm": "LinUCB Contextual Multi-Armed Bandit",
            "context": {
                "tenure_months": customer_data.get("tenure_months", 12),
                "churn_risk_pct": churn_probability,
                "estimated_ltv_usd": round(ltv_estimate, 2)
            }
        }

    def record_feedback(self, customer_data: dict, churn_probability: float, arm_id: int, observed_reward: float):
        if 0 <= arm_id < len(self.arms):
            x = self._extract_context(customer_data, churn_probability)
            self.A[arm_id] += np.dot(x, x.T)
            self.b[arm_id] += observed_reward * x
            return True
        return False

rl_service = ContextualBanditService()
