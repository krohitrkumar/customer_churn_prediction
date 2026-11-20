import asyncio
from typing import Optional
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, HTTPException, status, Depends, Request
from database.connection import db_dependency
from database.security import get_current_user, require_role
from middlewares.limiter import limiter
from models.user import User, UserRole
from models.customer import Customer
from models.prediction import PredictionHistory
from schemas.prediction import SinglePredictionRequest, PredictionResponse
from services.ml_services import ml_service
from services.shap_service import shap_service
from services.genai_service import genai_service
from services.rl_service import rl_service
from services.email_service import email_service

router = APIRouter(
    prefix="/predict",
    tags=["Predictions"]
)

class SendEmailPayload(BaseModel):
    recipient_email: str
    subject: str
    body: str

class RLFeedbackPayload(BaseModel):
    arm_id: int
    observed_reward: float
    tenure_months: int = 12
    support_calls: int = 0
    late_payments: int = 0
    satisfaction_score: float = 5.0
    churn_probability: float = 50.0

def get_user_identity(user: User):
    if not user:
        return "Account Specialist", "Customer Success Team"
    name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    if not name:
        name = user.email.split("@")[0].replace(".", " ").title()
    
    if user.role == UserRole.ADMIN:
        role = "Platform Administrator, Retentrix"
    elif user.role == UserRole.CSM:
        role = "Customer Success Specialist, Retentrix"
    else:
        role = "Account Specialist, Retentrix"
    return name, role

@router.post("/single", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
@limiter.limit("60/minute")
def predict_single_customer(
    request: Request,
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
            triggered_by_user_id=current_user.id,
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            playbook_recommendations=[p.model_dump() for p in result["playbooks"]]
        )
        db.add(history)
        db.commit()

    return result

@router.get("/copilot/{customer_id}", status_code=status.HTTP_200_OK)
def get_customer_retention_copilot(
    customer_id: int,
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} does not exist."
        )

    customer_dict = {
        "first_name": customer.first_name,
        "last_name": customer.last_name,
        "email": customer.email,
        "tenure_months": customer.tenure_months,
        "support_calls": customer.support_calls or 0,
        "late_payments": customer.late_payments or 0,
        "satisfaction_score": customer.satisfaction_score,
        "region": customer.region.value if hasattr(customer.region, "value") else str(customer.region),
        "contract_type": customer.contract_type.value if hasattr(customer.contract_type, "value") else str(customer.contract_type),
        "payment_method": customer.payment_method.value if hasattr(customer.payment_method, "value") else str(customer.payment_method)
    }

    if customer.latest_churn_score is None:
        prediction_result = ml_service.predict_churn(customer_dict)
        churn_prob = prediction_result["churn_probability"]
        customer.latest_churn_score = churn_prob
        customer.latest_risk_level = prediction_result["risk_level"]
        db.commit()
    else:
        churn_prob = float(customer.latest_churn_score)

    manager_name, manager_role = get_user_identity(current_user)
    shap_explanation = shap_service.explain_customer(customer_dict)
    copilot_plan = genai_service.generate_retention_copilot(
        customer_dict,
        churn_prob,
        shap_explanation,
        manager_name=manager_name,
        manager_role=manager_role
    )
    rl_decision = rl_service.select_action(customer_dict, churn_prob)

    return {
        "customer_id": customer.id,
        "customer_name": f"{customer.first_name} {customer.last_name}",
        "customer_code": customer.customer_code,
        "customer_email": customer.email,
        "churn_probability": churn_prob,
        "risk_level": customer.latest_risk_level,
        "shap_explanation": shap_explanation,
        "ai_copilot": copilot_plan,
        "rl_policy": rl_decision
    }

@router.post("/copilot", status_code=status.HTTP_200_OK)
def run_adhoc_copilot(
    payload: SinglePredictionRequest,
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):
    data = payload.model_dump(mode="json")
    cust_name = "Customer"
    cust_email = None

    if payload.customer_id:
        c = db.query(Customer).filter(Customer.id == payload.customer_id).first()
        if c:
            cust_name = f"{c.first_name} {c.last_name}".strip()
            cust_email = c.email
            data["first_name"] = c.first_name
            data["last_name"] = c.last_name

    prediction_result = ml_service.predict_churn(data)
    churn_prob = prediction_result["churn_probability"]
    manager_name, manager_role = get_user_identity(current_user)

    shap_explanation = shap_service.explain_customer(data)
    copilot_plan = genai_service.generate_retention_copilot(
        data,
        churn_prob,
        shap_explanation,
        manager_name=manager_name,
        manager_role=manager_role
    )
    rl_decision = rl_service.select_action(data, churn_prob)

    return {
        "customer_name": cust_name,
        "customer_email": cust_email,
        "churn_prediction": prediction_result["churn_prediction"],
        "churn_probability": churn_prob,
        "risk_level": prediction_result["risk_level"],
        "playbooks": prediction_result.get("playbooks", []),
        "shap_explanation": shap_explanation,
        "ai_copilot": copilot_plan,
        "rl_policy": rl_decision
    }

@router.post("/send-email", status_code=status.HTTP_200_OK)
async def send_retention_email(
    payload: SendEmailPayload,
    current_user: User = Depends(require_role(UserRole.ADMIN, UserRole.CSM))
):
    sender_name, _ = get_user_identity(current_user)
    success = await asyncio.to_thread(
        email_service.send_retention_email,
        payload.recipient_email,
        payload.subject,
        payload.body,
        sender_name=sender_name,
        reply_to=current_user.email
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to dispatch email. Please verify recipient email or SMTP settings."
        )
    return {"message": f"Retention email successfully delivered to {payload.recipient_email}"}

@router.post("/rl-feedback", status_code=status.HTTP_200_OK)
def record_rl_feedback(
    payload: RLFeedbackPayload,
    current_user: User = Depends(get_current_user)
):
    cust_data = {
        "tenure_months": payload.tenure_months,
        "support_calls": payload.support_calls,
        "late_payments": payload.late_payments,
        "satisfaction_score": payload.satisfaction_score
    }
    success = rl_service.record_feedback(
        cust_data,
        payload.churn_probability,
        payload.arm_id,
        payload.observed_reward
    )
    return {"status": "success", "updated_arm": payload.arm_id}

@router.get("/rl-policy/{customer_id}", status_code=status.HTTP_200_OK)
def get_customer_rl_policy(
    customer_id: int,
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} does not exist."
        )

    customer_dict = {
        "tenure_months": customer.tenure_months,
        "support_calls": customer.support_calls or 0,
        "late_payments": customer.late_payments or 0,
        "satisfaction_score": customer.satisfaction_score
    }
    churn_prob = float(customer.latest_churn_score) if customer.latest_churn_score is not None else 50.0

    return rl_service.select_action(customer_dict, churn_prob)

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
