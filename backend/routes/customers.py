from fastapi import APIRouter, HTTPException, status, Depends,UploadFile, File
from typing import List, Optional
import random
import io
import pandas as pd
from database.connection import db_dependency
from models.customer import Customer, ContractType, PaymentMethod, Region
from schemas.customer import CustomerCreate, CustomerUpdate, CustomerOut
from models.user import User, UserRole
from database.security import require_role, get_current_user
from models.prediction import PredictionHistory
from services.ml_services import ml_service

router = APIRouter(
    prefix="/customers", 
    tags=["Customers"]  
)

# 1. Get All Customers 
@router.get("/", response_model=List[CustomerOut], status_code=status.HTTP_200_OK)
def get_all_customers(
    db: db_dependency,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    customers = db.query(Customer).offset(skip).limit(limit).all()
    return customers

# 2. Create a Customer
@router.post("/", response_model=CustomerOut, status_code=status.HTTP_201_CREATED)
def create_customer(
    payload: CustomerCreate,
    db: db_dependency,
    current_user: User = Depends(require_role(UserRole.ADMIN, UserRole.CSM))
):
    existing = db.query(Customer).filter(Customer.customer_code == payload.customer_code).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Customer code '{payload.customer_code}' already exists."
        )
    new_customer = Customer(**payload.model_dump())
    db.add(new_customer)
    db.commit()
    db.refresh(new_customer)
    return new_customer

# 3. Get Customer by ID
@router.get("/{customer_id}", response_model=CustomerOut, status_code=status.HTTP_200_OK)
def get_customer_by_id(
    customer_id: int,
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found."
        )
    return customer

# 4. Update an Customer Profile
@router.put("/{customer_id}", response_model=CustomerOut, status_code=status.HTTP_200_OK)
def update_customer(
    customer_id: int, 
    payload: CustomerUpdate, 
    db: db_dependency,
    current_user: User = Depends(require_role(UserRole.ADMIN, UserRole.CSM))
):
    """Update customer details (e.g. support calls, satisfaction score, plan)."""
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Customer with ID {customer_id} not found."
        )

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(customer, field, value)

    db.commit()
    db.refresh(customer)
    return customer

# 5. Delete a Customer 
@router.delete("/{customer_id}", status_code=status.HTTP_200_OK)
def delete_customer(
    customer_id: int, 
    db: db_dependency,
    current_admin: User = Depends(require_role(UserRole.ADMIN))
):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Customer with ID {customer_id} not found."
        )
    db.delete(customer)
    db.commit()
    return {"message": f"Customer ID {customer_id} deleted successfully."}


@router.post("/seed", status_code=status.HTTP_200_OK)
def seed_sample_customers(
    db: db_dependency,
    count: int = 50,
    current_user: User = Depends(require_role(UserRole.ADMIN, UserRole.CSM))):

    first_names = [
        "Rohit","Mukul","Gopal","Tuhsar","Prachi","Ashish","Rahul","Mohit","Pushkar","Rishabh","Javed","Shubha","Vanshika","Aditi","Ankush","Aditya",
        "kunal","Kajol","krishna","Param","Shayam","Ram","Sushant","Sonam","Sahil","Sujal","Ashu","Shruti","Veshno","Ayushi","Anjali","Sushma"
    ]

    last_names = [
        "Kumar", "Yadav", "Sharma", "Singhal", "Gupta", "Bansal", "Rao", "Singh", "Reddy",
        "Trivedi", "Dube", "Joshi", "Khan", "Dhondhiyal", "Chaturvedi", "Iyer", "Jha", "Patel"
    ]

    contract_types = [ContractType.MONTH_TO_MONTH, ContractType.ONE_YEAR, ContractType.TWO_YEAR]
    payment_methods = [PaymentMethod.CARD, PaymentMethod.WALLET, PaymentMethod.BANK]
    regions = [Region.NORTH_AMERICA, Region.EUROPE, Region.ASIA, Region.LATAM, Region.AFRICA, Region.SOUTH_AMERICA]

    existing_code = set(code[0] for code in db.query(Customer.customer_code).all())
    created_count = 0

    for i in range(count):
        code_num = 1000+ i +len(existing_code)
        code = f"CUST-{code_num}"
        while code in existing_code:
            code_num+=1
            code = f"CUST-{code_num}"
        existing_code.add(code)

        fn = random.choice(first_names)
        ln = random.choice(last_names)
        email = f"{fn.lower()}.{ln.lower()}{random.randint(10,99)}@gmail.com"
        tenure = random.randint(1,60)
        contract = random.choice(contract_types)
        payment = random.choice(payment_methods)
        region = random.choice(regions)

        if contract == ContractType.MONTH_TO_MONTH:
            satisfaction = round(random.uniform(1.5, 7.5), 1)
            support_calls = random.choices([0, 1, 2, 4, 6, 8], weights=[15, 20, 25, 20, 10, 10])[0]
            late_payments = random.choices([0, 1, 2, 3, 5], weights=[30, 25, 20, 15, 10])[0]
        elif contract == ContractType.ONE_YEAR:
            satisfaction = round(random.uniform(4.0, 9.0), 1)
            support_calls = random.randint(0, 4)
            late_payments = random.choices([0, 1, 2], weights=[60, 30, 10])[0]
        else: # TWO_YEAR
            satisfaction = round(random.uniform(6.0, 10.0), 1)
            support_calls = random.randint(0, 2)
            late_payments = random.choices([0, 1], weights=[85, 15])[0]


        cust_data = {
            "customer_code": code,
            "first_name": fn,
            "last_name": ln,
            "email": email,
            "tenure_months": tenure,
            "support_calls": support_calls,
            "late_payments": late_payments,
            "satisfaction_score": satisfaction,
            "contract_type": contract,
            "payment_method": payment,
            "region": region
        }

        churn_score = None
        risk_level = None
        playbooks = []
        try:
            result = ml_service.predict_churn(cust_data)
            churn_score = result["churn_probability"]
            risk_level = result["risk_level"]
            playbooks = result.get("playbooks", [])
        except Exception:
            pass
        customer = Customer(
            **cust_data,
            latest_churn_score=churn_score,
            latest_risk_level=risk_level
        )
        db.add(customer)
        db.flush()
        
        if churn_score is not None:
            history = PredictionHistory(
                customer_id=customer.id,
                triggered_by_user_id=current_user.id,
                churn_probability=churn_score,
                risk_level=risk_level,
                playbook_recommendations=[p.model_dump() for p in playbooks] if playbooks else []
            )
            db.add(history)
        created_count += 1
    db.commit()
    return {"message": f"Successfully seeded {created_count} accounts with AI predictions."}

# File Uploader 

#cleaning 
def clean_int(val,default = 0 ,min_val = 0 ,max_val = 100):
    try:
        num = int(float(val)) if pd.notna(val) else default
        return max(min_val,min(max_val,num))
    except (ValueError,TypeError):
        return default

def clean_float(val, default=5.0, min_val=1.0, max_val=10.0) :
   
    try:
        num = float(val) if pd.notna(val) else default
        return round(max(min_val, min(max_val, num)), 1)
    except (ValueError, TypeError):
        return default
def clean_contract(val: str):
    v = str(val).lower().replace(" ", "_").replace("-", "_").strip()
    if any(k in v for k in ["two", "2_year", "24", "biennial", "two_year"]):
        return ContractType.TWO_YEAR
    if any(k in v for k in ["one", "1_year", "12", "annual", "yearly", "one_year"]):
        return ContractType.ONE_YEAR
    return ContractType.MONTH_TO_MONTH

def clean_payment(val: str) -> PaymentMethod:
    v = str(val).lower().strip().replace(" ", "_").replace("-", "_")
    if any(k in v for k in ["card", "credit", "debit", "visa", "mastercard", "amex"]):
        return PaymentMethod.CARD
    if any(k in v for k in ["wallet", "digital", "paypal", "apple", "google", "upi", "stripe"]):
        return PaymentMethod.WALLET
    if any(k in v for k in ["bank", "transfer", "wire", "ach", "direct_debit", "net_banking"]):
        return PaymentMethod.BANK
    return PaymentMethod.CARD

def clean_region(val: str) -> Region:
    v = str(val).lower().replace(" ", "_").replace("-", "_").strip()
    valid_regions = {r.value: r for r in Region}
    return valid_regions.get(v, Region.NORTH_AMERICA)

@router.post("/upload_file", status_code=status.HTTP_200_OK)
async def upload_customers(
    db: db_dependency,
    file: UploadFile = File(...),
    current_user: User = Depends(require_role(UserRole.ADMIN, UserRole.CSM))
):
    filename = file.filename.lower()
    contents = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format! Please upload a .csv, .xlsx, or .xls file."
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read spreadsheet: {str(e)}"
        )

    raw_cols = [str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]
    alias_map = {
        "customer_code": ["customer_code", "customer_id", "cust_id", "userid", "user_id", "id", "account_id", "code"],
        "first_name": ["first_name", "firstname", "first", "fname"],
        "last_name": ["last_name", "lastname", "last", "lname", "surname"],
        "full_name": ["name", "full_name", "customer_name", "client_name"],
        "tenure_months": ["tenure_months", "tenure", "tenuremonths", "months", "subscription_length", "duration"],
        "satisfaction_score": ["satisfaction_score", "satisfaction", "csat", "score", "rating", "nps"],
        "support_calls": ["support_calls", "calls", "tickets", "complaints", "inquiries"],
        "late_payments": ["late_payments", "late", "delayed_payments", "overdue"],
        "contract_type": ["contract_type", "contract", "plan", "subscription", "term", "plan_type"],
        "payment_method": ["payment_method", "payment", "method", "billing", "pay_method"],
        "region": ["region", "location", "country", "zone", "geo"],
        "email": ["email", "mail", "email_address"]
    }

    rename_dict = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias in raw_cols:
                original_col = df.columns[raw_cols.index(alias)]
                rename_dict[original_col] = canonical
                break

    df.rename(columns=rename_dict, inplace=True)

    if "full_name" in df.columns and "first_name" not in df.columns:
        names = df["full_name"].astype(str).str.strip().str.split(" ", n=1, expand=True)
        df["first_name"] = names[0]
        df["last_name"] = names[1] if names.shape[1] > 1 else ""

    if "first_name" not in df.columns:
        df["first_name"] = "Customer"
    if "last_name" not in df.columns:
        df["last_name"] = ""
    if "customer_code" not in df.columns:
        df["customer_code"] = [f"CUST-{random.randint(10000, 99999)}" for _ in range(len(df))]

    if "satisfaction_score" in df.columns:
        numeric_scores = pd.to_numeric(df["satisfaction_score"], errors="coerce").dropna()
        if len(numeric_scores) > 0 and numeric_scores.max() <= 5.0 and numeric_scores.max() > 1.0:
            df["satisfaction_score"] = df["satisfaction_score"] * 2.0
    existing_codes = set(code[0] for code in db.query(Customer.customer_code).all())
    valid_records = []
    skipped_count = 0

    for _, row in df.iterrows():
        code = str(row["customer_code"]).strip() if pd.notna(row.get("customer_code")) else ""
        if not code or code in existing_codes:
            skipped_count += 1
            continue

        existing_codes.add(code)
        fn = str(row["first_name"]).strip() if pd.notna(row.get("first_name")) else "Customer"
        ln = str(row["last_name"]).strip() if pd.notna(row.get("last_name")) else ""
        email = str(row["email"]).strip() if "email" in row and pd.notna(row["email"]) else None
        tenure = clean_int(row.get("tenure_months"), default=12, min_val=1, max_val=72)
        satisfaction = clean_float(row.get("satisfaction_score"), default=5.0, min_val=1.0, max_val=10.0)
        calls = clean_int(row.get("support_calls"), default=0, min_val=0, max_val=20)
        late = clean_int(row.get("late_payments"), default=0, min_val=0, max_val=12)
        contract = clean_contract(row.get("contract_type", "month_to_month"))
        payment = clean_payment(row.get("payment_method", "card"))
        region = clean_region(row.get("region", "north_america"))

        valid_records.append({
            "customer_code": code,
            "first_name": fn,
            "last_name": ln,
            "email": email,
            "tenure_months": tenure,
            "support_calls": calls,
            "late_payments": late,
            "satisfaction_score": satisfaction,
            "contract_type": contract,
            "payment_method": payment,
            "region": region
        })

    imported_count = len(valid_records)

    if imported_count > 0:
        batch_df = pd.DataFrame([{
            "tenure_months": r["tenure_months"],
            "support_calls": r["support_calls"],
            "late_payments": r["late_payments"],
            "satisfaction_score": r["satisfaction_score"],
            "region": r["region"].value,
            "contract_type": r["contract_type"].value,
            "payment_method": r["payment_method"].value
        } for r in valid_records])

        try:
            _, probabilities, risk_levels = ml_service.predict_churn_batch(batch_df)
        except Exception:
            probabilities = [None] * imported_count
            risk_levels = [None] * imported_count

        customers_to_add = []
        for i, record in enumerate(valid_records):
            prob = float(probabilities[i]) if probabilities[i] is not None else None
            risk = str(risk_levels[i]) if risk_levels[i] is not None else None

            customer = Customer(
                **record,
                latest_churn_score=prob,
                latest_risk_level=risk
            )
            customers_to_add.append(customer)

        db.add_all(customers_to_add)
        db.flush()

        histories_to_add = []
        for i, customer in enumerate(customers_to_add):
            prob = float(probabilities[i]) if probabilities[i] is not None else None
            risk = str(risk_levels[i]) if risk_levels[i] is not None else None
            if prob is not None:
                histories_to_add.append(PredictionHistory(
                    customer_id=customer.id,
                    triggered_by_user_id=current_user.id,
                    churn_probability=prob,
                    risk_level=risk,
                    playbook_recommendations=[]
                ))

        if histories_to_add:
            db.add_all(histories_to_add)

        db.commit()

    return {
        "message": f"Successfully imported {imported_count} accounts with AI churn predictions ({skipped_count} skipped).",
        "imported": imported_count,
        "skipped": skipped_count,
        "total_rows": len(df)
    }