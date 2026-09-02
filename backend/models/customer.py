import enum
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base

class ContractType(str, enum.Enum):
    MONTH_TO_MONTH = "month_to_month"
    ONE_YEAR = "one_year"
    TWO_YEAR = "two_year"

class PaymentMethod(str, enum.Enum):
    CARD = "card"
    WALLET = "wallet"
    BANK = "bank"

class Region(str, enum.Enum):
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA = "asia"
    LATAM = "latam"
    AFRICA = "africa"
    SOUTH_AMERICA = "south_america"

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    customer_code = Column(String(50), unique=True, index=True, nullable=False) 
    first_name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    
    tenure_months = Column(Integer, nullable=False)
    support_calls = Column(Integer, default=0)
    late_payments = Column(Integer, default=0)
    satisfaction_score = Column(Float, nullable=False)
    contract_type = Column(Enum(ContractType), nullable=False)
    payment_method = Column(Enum(PaymentMethod), nullable=False)
    region = Column(Enum(Region), nullable=False)
    
   
    latest_churn_score = Column(Float, nullable=True)
    latest_risk_level = Column(String(50), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


    predictions = relationship("PredictionHistory", back_populates="customer", cascade="all, delete-orphan")