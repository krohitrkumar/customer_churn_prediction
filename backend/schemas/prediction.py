from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from models.customer import ContractType, PaymentMethod, Region

class SinglePredictionRequest(BaseModel):
    tenure_months: int = Field(..., ge=1, le=100, examples=[12])
    support_calls: int = Field(..., ge=0, le=50, examples=[2])
    late_payments: int = Field(..., ge=0, le=30, examples=[1])
    satisfaction_score: float = Field(..., ge=1.0, le=10.0, examples=[5.0])
    contract_type: ContractType
    payment_method: PaymentMethod
    region: Region
    customer_id: Optional[int] = None

class PlaybookRecommendation(BaseModel):
    icon: str
    category: str
    action: str


class PredictionResponse(BaseModel):
    churn_prediction: int  
    churn_probability: float  
    risk_level: Literal["Low", "Moderate", "Critical"]
    playbooks: List[PlaybookRecommendation]