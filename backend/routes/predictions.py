from fastapi import APIRouter, HTTPException, status, Depends
from typing import List

from database.connection import db_dependency
from database.security import get_current_user
from models.user import User
from schemas.prediction import SinglePredictionRequest, PredictionResponse
from models.prediction import PredictionHistory
from services.ml_services import ml_service
from models.customer import Customer

router = APIRouter(
    prefix="/predict",
    tags=["Predictions"]
)

@router.post("/single", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_single_customer(
    payload: SinglePredictionRequest,
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):

    result = ml_service.predict_churn(payload.model_dump(mode="json"))
    if payload.customer_id:
        customer = db.query(Customer).filter(Customer.id == payload.customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer with ID {payload.customer_id} does not exist."
            )
        customer.latest_churn_score = result["churn_probability"]
        customer.latest_risk_level = result["risk_level"]

        history = PredictionHistory(
            customer_id=customer.id,
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            playbook_recommendations=[p.model_dump() for p in result["playbooks"]]
        )
        db.add(history)
        db.commit()

    return result

@router.get("/history/{customer_id}", status_code=status.HTTP_200_OK)
def get_customer_prediction_history(
    customer_id: int,
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):

    history_records = db.query(PredictionHistory).filter(
        PredictionHistory.customer_id == customer_id
    ).order_by(PredictionHistory.created_at.desc()).all()

    if not history_records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"No prediction history found for customer ID {customer_id}"
        )
    return history_records



               