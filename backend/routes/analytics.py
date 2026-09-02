from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import func, case
from typing import List
from database.connection import db_dependency
from database.security import get_current_user
from models.user import User
from models.customer import Customer, ContractType, PaymentMethod, Region
from models.prediction import PredictionHistory

from schemas.analytics import (
    AnalyticsSummary,
    RiskBreakdownResponse,
    CategoryBreakdownItem,
    TopRiskCustomer,
    RecentActivityItem
)

router = APIRouter(
    prefix="/analytics",
    tags=["Executive Analytics"]
)



@router.get("/summary",response_model=AnalyticsSummary,status_code=status.HTTP_200_OK)
def get_analytics_summary(
    db: db_dependency,
    current_user:User = Depends(get_current_user)
):
    total_customers = db.query(func.count(Customer.id)).scalar() or 0
    scored_customers = db.query(func.count(Customer.id)).filter(Customer.latest_churn_score.isnot(None)).scalar() or 0
    avg_churn = db.query(func.avg(Customer.latest_churn_score)).filter(Customer.latest_churn_score.isnot(None)).scalar() or 0.0
    avg_satisfaction = db.query(func.avg(Customer.satisfaction_score)).scalar() or 0.0
    critical_count = db.query(func.count(Customer.id)).filter(Customer.latest_risk_level == "Critical").scalar() or 0
    moderate_count = db.query(func.count(Customer.id)).filter(Customer.latest_risk_level == "Moderate").scalar() or 0
    low_count = db.query(func.count(Customer.id)).filter(Customer.latest_risk_level == "Low").scalar() or 0
    unscored_count = total_customers - scored_customers
    critical_rate = round((critical_count / scored_customers * 100), 2) if scored_customers > 0 else 0.0

    return AnalyticsSummary(
        total_customers=total_customers,
        scored_customers=scored_customers,
        avg_churn_score=round(float(avg_churn), 4),
        avg_satisfaction_score=round(float(avg_satisfaction), 2),
        critical_risk_count=critical_count,
        moderate_risk_count=moderate_count,
        low_risk_count=low_count,
        unscored_count=unscored_count,
        critical_rate_pct=critical_rate
    )

@router.get("/risk_breakdown", response_model=RiskBreakdownResponse, status_code=status.HTTP_200_OK)
def get_risk_breakdown(
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):
    def aggregate_by_column(column):
        results = db.query(
            column,
            func.count(Customer.id).label("total"),
            func.sum(case((Customer.latest_risk_level == "Critical", 1), else_=0)).label("critical"),
            func.sum(case((Customer.latest_risk_level == "Moderate", 1), else_=0)).label("moderate"),
            func.sum(case((Customer.latest_risk_level == "Low", 1), else_=0)).label("low"),
            func.avg(Customer.latest_churn_score).label("avg_churn")
        ).group_by(column).all()
        items = []
        for row in results:
            val_name = row[0].value if hasattr(row[0], "value") else str(row[0])
            items.append(CategoryBreakdownItem(
                category=val_name.replace("_", " ").title() if val_name else "Unknown",
                total=row.total or 0,
                critical=int(row.critical or 0),
                moderate=int(row.moderate or 0),
                low=int(row.low or 0),
                avg_churn_score=round(float(row.avg_churn or 0), 4)
            ))
        return items
    return RiskBreakdownResponse(
        by_contract=aggregate_by_column(Customer.contract_type),
        by_region=aggregate_by_column(Customer.region),
        by_payment=aggregate_by_column(Customer.payment_method)
    )

@router.get("/top_at_risk", response_model=List[TopRiskCustomer], status_code=status.HTTP_200_OK)
def get_top_at_risk_customers(
    db: db_dependency,
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    customers = db.query(Customer).filter(
        Customer.latest_risk_level == "Critical"
    ).order_by(Customer.latest_churn_score.desc()).limit(limit).all()

    return [
        TopRiskCustomer(
            id=c.id,
            customer_code=c.customer_code,
            first_name=c.first_name,
            last_name=c.last_name,
            email=c.email,
            tenure_months=c.tenure_months,
            satisfaction_score=c.satisfaction_score,
            support_calls=c.support_calls or 0,
            late_payments=c.late_payments or 0,
            contract_type=c.contract_type.value if hasattr(c.contract_type, "value") else str(c.contract_type),
            region=c.region.value if hasattr(c.region, "value") else str(c.region),
            latest_churn_score=c.latest_churn_score or 0.0,
            latest_risk_level=c.latest_risk_level or "Critical"
        )
        for c in customers
    ]

@router.get("/recent_activity", response_model=List[RecentActivityItem], status_code=status.HTTP_200_OK)
def get_recent_activity(
    db: db_dependency,
    limit: int = 15,
    current_user: User = Depends(get_current_user)
):
    records = db.query(PredictionHistory).order_by(
        PredictionHistory.created_at.desc()
    ).limit(limit).all()
    items = []
    for r in records:
        cust = r.customer
        items.append(RecentActivityItem(
            id=r.id,
            customer_id=r.customer_id,
            customer_name=f"{cust.first_name} {cust.last_name}" if cust else "Direct Prediction",
            customer_code=cust.customer_code if cust else "N/A",
            churn_probability=r.churn_probability,
            risk_level=r.risk_level,
            created_at=r.created_at
        ))
    return items