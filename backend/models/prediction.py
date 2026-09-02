# backend/models/prediction.py
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)
    triggered_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    churn_probability = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    playbook_recommendations = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    customer = relationship("Customer", back_populates="predictions")
    triggered_by_user = relationship("User", back_populates="predictions_created")