from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from models.customer import ContractType, PaymentMethod, Region

class CustomerBase(BaseModel):
    customer_code: str = Field(..., examples=["CUST-1001"])
    first_name: str = Field(..., examples=["John"])
    last_name: str = Field(..., examples=["Doe"])
    email: Optional[str] = Field(None, examples=["john.doe@example.com"])
    tenure_months: int = Field(0, ge=0, le=100, description="Tenure in Months", examples=[14])
    support_calls: int = Field(0, ge=0, le=50, description="Support Ticket count", examples=[2])
    late_payments: int = Field(0, ge=0, le=30, description="Late payment count", examples=[1])
    satisfaction_score: float = Field(5.0, ge=0.0, le=10.0, description="Score 0-10", examples=[6.5])
    contract_type: ContractType
    payment_method: PaymentMethod
    region: Region

class CustomerCreate(CustomerBase):
    pass

class CustomerUpdate(BaseModel):
    customer_code: Optional[str] = None
    first_name: Optional[str] = None 
    last_name: Optional[str] = None
    email: Optional[str] = None
    tenure_months: Optional[int] = Field(None, ge=0, le=100)
    support_calls: Optional[int] = Field(None, ge=0, le=50)
    late_payments: Optional[int] = Field(None, ge=0, le=30)
    satisfaction_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    contract_type: Optional[ContractType] = None
    payment_method: Optional[PaymentMethod] = None
    region: Optional[Region] = None


class CustomerOut(CustomerBase):
    id: int
    latest_churn_score: Optional[float] = None
    latest_risk_level: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True