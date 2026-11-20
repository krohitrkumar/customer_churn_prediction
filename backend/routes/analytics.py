from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy import func, case
from sqlalchemy.orm import joinedload
from database.connection import db_dependency
from database.security import get_current_user
from models.user import User
from models.customer import Customer
from models.prediction import PredictionHistory
from services.cache_service import cache_service
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

@router.get("/summary", response_model=AnalyticsSummary, status_code=status.HTTP_200_OK)
def get_analytics_summary(
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):
    cached_summary = cache_service.get_json("analytics_summary")
    if cached_summary:
        return AnalyticsSummary(**cached_summary)

    agg = db.query(
        func.count(Customer.id).label("total"),
        func.count(Customer.latest_churn_score).label("scored"),
        func.avg(Customer.latest_churn_score).label("avg_churn"),
        func.avg(Customer.satisfaction_score).label("avg_satisfaction"),
        func.sum(case((Customer.latest_risk_level == "Critical", 1), else_=0)).label("critical"),
        func.sum(case((Customer.latest_risk_level == "Moderate", 1), else_=0)).label("moderate"),
        func.sum(case((Customer.latest_risk_level == "Low", 1), else_=0)).label("low")
    ).first()

    total_customers = agg.total or 0
    scored_customers = agg.scored or 0
    avg_churn = float(agg.avg_churn or 0.0)
    avg_satisfaction = float(agg.avg_satisfaction or 0.0)
    critical_count = int(agg.critical or 0)
    moderate_count = int(agg.moderate or 0)
    low_count = int(agg.low or 0)
    unscored_count = total_customers - scored_customers
    critical_rate = round((critical_count / scored_customers * 100), 2) if scored_customers > 0 else 0.0

    avg_ltv = 300.0 + (18.0 * 24.0)
    arr_at_risk = round(critical_count * avg_ltv * (avg_churn / 100.0), 2)
    saved_revenue = round(arr_at_risk * 0.65, 2)

    summary_data = {
        "total_customers": total_customers,
        "scored_customers": scored_customers,
        "avg_churn_score": round(avg_churn, 4),
        "avg_satisfaction_score": round(avg_satisfaction, 2),
        "critical_risk_count": critical_count,
        "moderate_risk_count": moderate_count,
        "low_risk_count": low_count,
        "unscored_count": unscored_count,
        "critical_rate_pct": critical_rate,
        "estimated_arr_at_risk_usd": arr_at_risk,
        "projected_ai_savings_usd": saved_revenue,
        "avg_customer_ltv_usd": avg_ltv
    }

    cache_service.set_json("analytics_summary", summary_data, expire_seconds=60)
    return AnalyticsSummary(**summary_data)

@router.get("/roi_analysis", status_code=status.HTTP_200_OK)
def get_roi_analysis(
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):
    cached_roi = cache_service.get_json("roi_analysis")
    if cached_roi:
        return cached_roi

    customers = db.query(Customer).filter(Customer.latest_churn_score.isnot(None)).all()

    heavy_lifters = []
    high_drain = []
    total_arr_at_risk = 0.0
    total_potential_saved = 0.0

    for c in customers:
        prob = float(c.latest_churn_score or 0.0) / 100.0
        ltv = 300.0 + (c.tenure_months * 18.0)
        at_risk_amount = ltv * prob

        if prob > 0.40:
            total_arr_at_risk += at_risk_amount
            total_potential_saved += at_risk_amount * 0.65

        c_obj = {
            "id": c.id,
            "customer_code": c.customer_code,
            "name": f"{c.first_name} {c.last_name}".strip(),
            "tenure_months": c.tenure_months,
            "contract_type": c.contract_type.value if hasattr(c.contract_type, "value") else str(c.contract_type),
            "churn_probability": round(float(c.latest_churn_score or 0.0), 1),
            "estimated_ltv_usd": round(ltv, 2),
            "support_calls": c.support_calls or 0,
            "satisfaction_score": c.satisfaction_score
        }

        if c.tenure_months >= 20 and prob > 0.35:
            heavy_lifters.append(c_obj)
        elif (c.support_calls or 0) >= 5 and c.satisfaction_score <= 4.0:
            high_drain.append(c_obj)

    heavy_lifters.sort(key=lambda x: x["estimated_ltv_usd"], reverse=True)
    high_drain.sort(key=lambda x: x["support_calls"], reverse=True)

    result_data = {
        "total_arr_at_risk_usd": round(total_arr_at_risk, 2),
        "total_projected_savings_usd": round(total_potential_saved, 2),
        "avg_customer_ltv_usd": 732.0,
        "heavy_lifters": heavy_lifters[:6],
        "high_drain_accounts": high_drain[:6],
        "what_if_scenarios": {
            "reduce_5pct": round(total_arr_at_risk * 0.15, 2),
            "reduce_10pct": round(total_arr_at_risk * 0.30, 2),
            "reduce_20pct": round(total_arr_at_risk * 0.55, 2)
        }
    }

    cache_service.set_json("roi_analysis", result_data, expire_seconds=60)
    return result_data

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
    records = db.query(PredictionHistory).options(
        joinedload(PredictionHistory.customer)
    ).order_by(
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
