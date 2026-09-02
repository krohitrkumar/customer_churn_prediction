from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class AnalyticsSummary(BaseModel):
    total_customers: int
    scored_customers: int
    avg_churn_score: float
    avg_satisfaction_score: float   
    critical_risk_count: int        
    moderate_risk_count: int        
    low_risk_count: int
    unscored_count: int
    critical_rate_pct: float

class CategoryBreakdownItem(BaseModel):
    category: str
    total: int
    critical: int
    moderate: int
    low: int
    avg_churn_score: float

class RiskBreakdownResponse(BaseModel):
    by_contract: List[CategoryBreakdownItem]
    by_region: List[CategoryBreakdownItem]
    by_payment: List[CategoryBreakdownItem]

class TopRiskCustomer(BaseModel):
    id: int
    customer_code: str
    first_name: str
    last_name: str
    email: Optional[str] = None
    tenure_months: int
    satisfaction_score: float
    support_calls: int
    late_payments: int
    contract_type: str
    region: str
    latest_churn_score: float
    latest_risk_level: str


class RecentActivityItem(BaseModel):
    id: int
    customer_id: Optional[int] = None
    customer_name: Optional[str] = None
    customer_code: Optional[str] = None
    churn_probability: float
    risk_level: str
    created_at: Optional[datetime] = None