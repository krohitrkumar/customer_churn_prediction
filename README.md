# 🧠 Customer Churn Intelligence Platform

> A production-grade, full-stack AI system that predicts customer churn in real time, manages enterprise accounts, and delivers actionable retention playbooks — powered by a Gradient Boosting ML model served via a secure REST API.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red)](https://sqlalchemy.org)

---

## 📑 Table of Contents

1. [What This Does](#-what-this-does)
2. [System Architecture](#-system-architecture)
3. [Project Structure](#-project-structure)
4. [ML Pipeline](#-ml-pipeline)
5. [Backend — FastAPI](#-backend--fastapi)
   - [Database Models](#database-models)
   - [API Endpoints](#api-endpoints)
   - [Authentication & Security](#authentication--security)
   - [Services](#services)
6. [Frontend — React](#-frontend--react)
7. [Feature Schema](#-feature-schema--model-inputs)
8. [Setup & Running Locally](#-setup--running-locally)
9. [Environment Variables](#-environment-variables)
10. [Business Impact](#-business-impact)

---

## 🎯 What This Does

This platform gives Customer Success Managers (CSMs) and Admins a **real-time churn intelligence cockpit**:

- 📊 **Predict** the probability of a customer cancelling their subscription using a trained ML model
- 🏢 **Manage** an enterprise customer database — create, update, delete, bulk-import via CSV/Excel
- ⚠️ **Identify** which accounts are at Critical / Moderate / Low risk
- 📋 **Serve retention playbooks** — rule-based automated recommendations triggered by the customer profile
- 📈 **Visualize** analytics across contract types, geographies, and payment methods
- 🔐 **Secure** every action behind JWT authentication with role-based access control (RBAC)

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                          CLIENT BROWSER                           │
│                                                                   │
│   React 18 + Vite  ──── Axios ────► FastAPI REST API             │
│   (SPA, JWT auth,                   (port 8000)                   │
│    role-based UI)                                                 │
└───────────────────────────┬───────────────────────────────────────┘
                            │
            ┌───────────────▼───────────────┐
            │        FastAPI Backend         │
            │                               │
            │  ┌─────────────────────────┐  │
            │  │      Route Handlers     │  │
            │  │  /auth  /customers      │  │
            │  │  /predict  /analytics   │  │
            │  └────────────┬────────────┘  │
            │               │               │
            │  ┌────────────▼────────────┐  │
            │  │     Services Layer      │  │
            │  │  MLService  EmailService│  │
            │  └────────────┬────────────┘  │
            │               │               │
            │  ┌────────────▼────────────┐  │
            │  │    SQLAlchemy ORM       │  │
            │  │  PostgreSQL / SQLite    │  │
            │  └─────────────────────────┘  │
            │                               │
            │  ┌─────────────────────────┐  │
            │  │   ML Artifact (.pkl)    │  │
            │  │  GradientBoostingPipeline│ │
            │  │  serialized via Joblib  │  │
            │  └─────────────────────────┘  │
            └───────────────────────────────┘
```

**Data flow for a churn prediction:**
1. Frontend sends customer feature data to `POST /api/predict/single`
2. Backend validates JWT → checks role
3. `MLService.predict_churn()` builds a pandas DataFrame and runs it through the serialized pipeline
4. Pipeline applies `ColumnTransformer` (OneHotEncoder for categoricals) → `GradientBoostingClassifier`
5. Returns `churn_probability` (0–100 scale), `risk_level`, and a list of `playbook_recommendations`
6. Result is persisted to `prediction_history` and written back to the `customers` table
7. Response sent to the React frontend for display

---

## 📁 Project Structure

```
customer_churn_prediction/
│
├── backend/                          # FastAPI application
│   ├── main.py                       # App factory, CORS middleware, router mounting
│   ├── .env                          # Environment configuration (gitignored)
│   │
│   ├── database/
│   │   ├── config.py                 # Pydantic Settings — all environment variables
│   │   ├── connection.py             # SQLAlchemy engine, session factory, db_dependency
│   │   └── security.py              # bcrypt hashing, JWT creation/decoding, RBAC guards
│   │
│   ├── models/                       # SQLAlchemy ORM table definitions
│   │   ├── customer.py               # Customer, ContractType, PaymentMethod, Region enums
│   │   ├── user.py                   # User, UserRole, OTPVerification
│   │   └── prediction.py            # PredictionHistory
│   │
│   ├── schemas/                      # Pydantic request/response schemas
│   │   ├── auth.py
│   │   ├── customer.py
│   │   ├── prediction.py
│   │   └── analytics.py
│   │
│   ├── routes/                       # FastAPI route handlers
│   │   ├── auth.py                   # Register, Login, OTP, Change/Reset Password
│   │   ├── customers.py              # CRUD, Seed, Bulk CSV/Excel Upload with ML scoring
│   │   ├── predictions.py            # Single prediction endpoint + history
│   │   └── analytics.py             # Summary KPIs, risk breakdown, top-at-risk, activity
│   │
│   ├── services/
│   │   ├── ml_services.py            # MLService: loads model, runs inference, generates playbooks
│   │   └── email_service.py          # SMTP email delivery for OTP codes (HTML template)
│   │
│   └── middlewares/
│       └── timing.py                 # ProcessTimeMiddleware — adds X-Process-Time header
│
├── frontend/                         # React 18 + Vite SPA
│   ├── src/
│   │   ├── api/                      # Axios API layer
│   │   │   ├── client.js             # Axios instance, JWT interceptor, 401 redirect
│   │   │   ├── auth.js               # Auth calls (login, register, OTP, password)
│   │   │   ├── customers.js          # Customer CRUD + upload (5-min timeout) + seed
│   │   │   ├── analytics.js          # Analytics API calls
│   │   │   └── predictions.js        # Prediction API calls
│   │   │
│   │   ├── context/
│   │   │   └── AuthContext.jsx       # Global auth state, role helpers, token management
│   │   │
│   │   ├── hooks/
│   │   │   ├── useCustomers.js       # Module-level 30s cache — no re-fetch on navigation
│   │   │   └── useToast.js           # Toast notification state manager
│   │   │
│   │   ├── pages/
│   │   │   ├── LoginPage.jsx         # Email/password, OTP modal, forgot-password modal
│   │   │   ├── RegisterPage.jsx      # New user registration form
│   │   │   ├── OtpPage.jsx           # Passwordless email OTP sign-in
│   │   │   ├── DashboardPage.jsx     # Executive KPIs, donut chart, segment tabs, activity
│   │   │   ├── CustomersPage.jsx     # Customer table, bulk CSV upload, seed, risk tabs
│   │   │   ├── CustomerDetailPage.jsx# Account detail + churn gauge + prediction history
│   │   │   ├── PredictPage.jsx       # On-demand single prediction form
│   │   │   └── SettingsPage.jsx      # Profile + two password-reset methods
│   │   │
│   │   └── components/
│   │       ├── dashboard/            # KpiCard, RiskPieChart (interactive), ChurnGauge
│   │       ├── customers/            # CustomerForm, RiskBadge
│   │       ├── layout/               # Sidebar, TopBar, Layout wrapper
│   │       ├── auth/                 # OtpInput (6-box slots)
│   │       └── ui/                   # Button, Badge, Modal, Toast, Skeleton
│   │
│   ├── package.json
│   └── vite.config.js               # Dev proxy: /api → http://localhost:8000
│
├── ml_pipeline/                      # Offline training pipeline
│   ├── train.py                      # Trains and exports churn_model.pkl
│   ├── data/
│   │   └── processed.csv             # Labelled training dataset
│   ├── artifacts/
│   │   └── churn_model.pkl           # Serialized Scikit-Learn pipeline (loaded by backend)
│   └── notebooks/
│       └── customer_churn.ipynb      # Exploratory data analysis notebook
│
└── requirements.txt                  # All Python backend + ML dependencies
```

---

## 🤖 ML Pipeline

### Model Architecture

The ML artifact is a **Scikit-Learn Pipeline** with two sequential stages:

```
Input DataFrame (7 features)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  ColumnTransformer  (Preprocessing stage)         │
│                                                   │
│  Categorical → OneHotEncoder                      │
│    drop='first', handle_unknown='ignore'          │
│    • contract_type   → 2 binary columns           │
│    • payment_method  → 2 binary columns           │
│    • region          → 5 binary columns           │
│                                                   │
│  Numerical → passthrough (no scaling)             │
│    • tenure_months, support_calls,                │
│      late_payments, satisfaction_score            │
└────────────────────────┬──────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────┐
│  GradientBoostingClassifier                       │
│    n_estimators   = 200   learning_rate = 0.05    │
│    max_depth      = 5     random_state  = 42      │
│    min_samples_split = 5  min_samples_leaf = 2    │
└───────────────────────────────────────────────────┘
        │
        ▼
   predict_proba → probability ×100 → stored as 0–100 float
   predict       → binary label (0 = stay, 1 = churn)
```

### Risk Level Thresholds

| Churn Probability | Risk Level |
|---|---|
| > 75% | 🔴 **Critical** |
| 40% – 75% | 🟠 **Moderate** |
| < 40% | 🟢 **Low** |

### Retention Playbook Engine (Rule-Based Post-Processing)

After every prediction, the backend evaluates five independent trigger conditions and attaches relevant playbooks to the response:

| Trigger Condition | Playbook Category | Recommended Action |
|---|---|---|
| `satisfaction_score ≤ 3.0` | Customer Satisfaction | Trigger immediate executive outreach + customer survey |
| `support_calls ≥ 5` | Support Queue Priority | Assign a Senior Technical Specialist to open tickets |
| `late_payments ≥ 3` | Billing Flexibility | Offer payment restructuring or automated installment reminders |
| `tenure_months < 12` | Onboarding Retention | Enroll in high-touch onboarding check-in call program |
| `contract_type == month_to_month` | Contract Commitment | Offer 15% discount incentive for annual plan upgrade |
| *(no conditions met)* | Healthy Account | Maintain standard quarterly automated product updates |

### Training & Retraining

```bash
python ml_pipeline/train.py
```

Outputs evaluation metrics to console and overwrites `ml_pipeline/artifacts/churn_model.pkl`.

---

## ⚙️ Backend — FastAPI

### Database Models

#### `users` table

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `email` | String(255) unique | Login identifier |
| `first_name` | String(255) | Optional |
| `last_name` | String(255) | Optional |
| `hashed_password` | String(255) | bcrypt hash |
| `role` | Enum | `admin` / `csm` / `user` |
| `is_active` | Boolean | Default `true` |
| `created_at` | DateTime (TZ) | Server default |

#### `otp_verifications` table

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | |
| `email` | String(255) | Target address |
| `otp_code` | String(6) | 6-digit random code |
| `expires_at` | DateTime (TZ) | 10 minutes from creation |
| `is_verified` | Boolean | Marked true after use |

#### `customers` table

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | |
| `customer_code` | String(50) unique | e.g. `CUST-1042` |
| `first_name` | String(255) | |
| `last_name` | String(255) | |
| `email` | String(255) nullable | |
| `tenure_months` | Integer | 1–72 |
| `support_calls` | Integer | Default 0 |
| `late_payments` | Integer | Default 0 |
| `satisfaction_score` | Float | 1.0–10.0 |
| `contract_type` | Enum | `month_to_month` / `one_year` / `two_year` |
| `payment_method` | Enum | `card` / `wallet` / `bank` |
| `region` | Enum | `north_america` / `europe` / `asia` / `latam` / `africa` / `south_america` |
| `latest_churn_score` | Float nullable | Last churn probability (0–100) |
| `latest_risk_level` | String(50) nullable | `Critical` / `Moderate` / `Low` |
| `created_at` | DateTime (TZ) | |
| `updated_at` | DateTime (TZ) | Auto-updates on edit |

#### `prediction_history` table

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | |
| `customer_id` | FK → customers.id | |
| `triggered_by_user_id` | FK → users.id | |
| `churn_probability` | Float | 0–100 |
| `risk_level` | String(50) | |
| `playbook_recommendations` | JSON | Array of playbook objects |
| `created_at` | DateTime (TZ) | |

---

### API Endpoints

All routes are prefixed with `/api`. Interactive docs at `/docs`.

#### 🔑 Auth — `/api/auth`

| Method | Path | Auth Required | Role | Description |
|---|---|---|---|---|
| `POST` | `/auth/register` | ❌ | — | Register a new user account |
| `POST` | `/auth/login` | ❌ | — | Login (form-data: username + password) → returns JWT |
| `GET` | `/auth/me` | ✅ JWT | any | Get current user profile |
| `POST` | `/auth/send_otp` | ❌ | — | Send 6-digit OTP to email (10-minute TTL) |
| `POST` | `/auth/verify_otp` | ❌ | — | Verify OTP — returns access token if user exists |
| `POST` | `/auth/change_password` | ✅ JWT | any | Change password (requires current password) |
| `POST` | `/auth/reset_password` | ❌ | — | Reset password via OTP (no login required) |

> **Login note:** Uses `application/x-www-form-urlencoded` with `username` (email) and `password` fields to comply with OAuth2PasswordRequestForm. All other endpoints accept JSON.

**OTP passwordless login flow:**
```
POST /auth/send_otp  { email }
POST /auth/verify_otp  { email, otp_code }  →  { access_token }
```

**Forgot password flow:**
```
POST /auth/send_otp  { email }
POST /auth/reset_password  { email, otp_code, new_password }
```

---

#### 👥 Customers — `/api/customers`

| Method | Path | Auth | Role | Description |
|---|---|---|---|---|
| `GET` | `/customers/` | ✅ | any | List customers (`skip`, `limit` pagination) |
| `POST` | `/customers/` | ✅ | admin, csm | Create a single customer |
| `GET` | `/customers/{id}` | ✅ | any | Get customer by ID |
| `PUT` | `/customers/{id}` | ✅ | admin, csm | Update customer profile |
| `DELETE` | `/customers/{id}` | ✅ | admin | Permanently delete customer |
| `POST` | `/customers/seed` | ✅ | admin, csm | Seed N demo accounts with live ML predictions |
| `POST` | `/customers/upload_file` | ✅ | admin, csm | Bulk import CSV/Excel with automatic ML scoring |

**Bulk Upload — Smart Processing:**
- Accepts `.csv`, `.xlsx`, `.xls`
- Column names are **case-insensitive** (lowercased + underscored before matching)
- **Required columns:** `customer_code`, `first_name`, `last_name`, `tenure_months`, `satisfaction_score`
- **Optional columns:** `email`, `support_calls`, `late_payments`, `contract_type`, `payment_method`, `region`
- Numeric values are clamped to valid model ranges; invalid rows are skipped gracefully
- Duplicate `customer_code` rows are skipped automatically
- Each new row gets a live ML prediction written to `prediction_history`
- Returns: `{ imported, skipped, total_rows, message }`

---

#### 🧠 Predictions — `/api/predict`

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/predict/single` | ✅ | Run ML prediction on a customer profile |
| `GET` | `/predict/history/{customer_id}` | ✅ | Full prediction history for a customer |

**Request body:**
```json
{
  "tenure_months": 6,
  "support_calls": 7,
  "late_payments": 3,
  "satisfaction_score": 2.5,
  "contract_type": "month_to_month",
  "payment_method": "card",
  "region": "europe",
  "customer_id": 42
}
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 87.34,
  "risk_level": "Critical",
  "playbooks": [
    { "icon": "alert-circle", "category": "Customer Satisfaction", "action": "Trigger immediate executive outreach..." },
    { "icon": "phone-call",   "category": "Support Queue Priority", "action": "Assign a Senior Technical Specialist..." }
  ]
}
```

If `customer_id` is provided, the result is persisted and the customer's `latest_churn_score` and `latest_risk_level` are updated.

---

#### 📊 Analytics — `/api/analytics`

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/analytics/summary` | ✅ | Executive KPI summary (totals, averages, risk counts) |
| `GET` | `/analytics/risk_breakdown` | ✅ | Risk distribution by contract type, region, payment method |
| `GET` | `/analytics/top_at_risk` | ✅ | Top N Critical-risk customers ordered by score |
| `GET` | `/analytics/recent_activity` | ✅ | Latest prediction history feed |

**Summary response fields:** `total_customers`, `scored_customers`, `avg_churn_score`, `avg_satisfaction_score`, `critical_risk_count`, `moderate_risk_count`, `low_risk_count`, `unscored_count`, `critical_rate_pct`

---

### Authentication & Security

| Mechanism | Implementation |
|---|---|
| Password hashing | `bcrypt` with per-password salt |
| JWT tokens | `python-jose` — HS256, 24-hour expiry |
| Token transport | `Authorization: Bearer <token>` header |
| Role enforcement | `require_role(*roles)` FastAPI dependency — raises 403 if role not in allowed list |
| OTP security | 6-digit random int, 10-minute TTL, single-use flag (`is_verified`) |
| Session expiry | Frontend clears localStorage and redirects on any 401 response |

**Roles and permissions:**

| Role | What they can do |
|---|---|
| `admin` | Everything — CRUD, delete, seed, upload, analytics, all routes |
| `csm` | Create/edit customers, upload, run predictions, view analytics |
| `user` | Read-only — view customers, run predictions, view analytics |

---

### Services

**`MLService`** (`backend/services/ml_services.py`)
- Loads `churn_model.pkl` once at server startup via `joblib`
- Lazy-reloads if artifact was missing at startup
- Converts SQLAlchemy enum values to raw strings before building the pandas DataFrame
- Returns `churn_probability` as a percentage value (0–100)

**`EmailService`** (`backend/services/email_service.py`)
- Sends OTP codes via SMTP using Python's stdlib `smtplib` (no extra dependency)
- Renders an HTML email template with the 6-digit code
- **Dev mode:** if `SMTP_USER` is not configured, prints the OTP to console instead of sending — safe for local development
- Configured via `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` env vars

**`ProcessTimeMiddleware`** (`backend/middlewares/timing.py`)
- Attaches `X-Process-Time` header to every response for latency monitoring

---

## ⚛️ Frontend — React

Built with **React 18 + Vite**. All API calls go through a centralized Axios client.

### Pages

| Page | Route | Description |
|---|---|---|
| Login | `/login` | Email/password, OTP login modal, 2-step forgot-password modal |
| Register | `/register` | New account creation |
| OTP Sign-in | `/otp` | Passwordless sign-in via emailed 6-digit code |
| Dashboard | `/dashboard` | KPI cards, interactive risk donut chart, segment breakdown tabs, recent activity feed |
| Customers | `/customers` | Searchable, sortable customer table; risk filter tabs; bulk upload; seed |
| Customer Detail | `/customers/:id` | Full profile, SVG churn gauge, prediction history timeline |
| Predict | `/predict` | On-demand prediction form for any customer |
| Settings | `/settings` | Profile + Method 1 (old password) + Method 2 (OTP reset) |

### Key Architecture Decisions

| Decision | Why |
|---|---|
| **Module-level 30s cache in `useCustomers`** | Prevents re-fetching the customer list on every page navigation; all components share the same in-memory snapshot |
| **Separate `uploadClient` (5-min timeout)** | Bulk upload of 1000+ rows each needing ML inference can take several minutes; the default 15s timeout would false-positive |
| **Client-side column validation before upload** | Reads CSV first line locally, detects missing required columns, shows an inline error without making any network request |
| **Smart upload result messages** | Backend returns `{ imported, skipped, total_rows }`; frontend generates "All 1200 already existed" vs "3 of 50 imported" |
| **Analytics fetched once on mount** | `useEffect` dependency is `[]` — dashboard analytics don't change just because customer count changes |

---

## 📐 Feature Schema & Model Inputs

| Feature | Type | Valid Range | Business Meaning |
|---|---|---|---|
| `tenure_months` | `int` | 1–72 | Months as an active subscriber — longer tenure = lower churn risk |
| `support_calls` | `int` | 0–20 | Helpdesk contact volume — high calls signal customer friction |
| `late_payments` | `int` | 0–12 | Delayed invoice count — positively correlated with churn |
| `satisfaction_score` | `float` | 1.0–10.0 | Customer satisfaction rating — ≤ 3.0 triggers executive outreach playbook |
| `contract_type` | `enum` | month_to_month / one_year / two_year | Short-term contracts show highest churn velocity |
| `payment_method` | `enum` | card / wallet / bank | Invoice payment channel |
| `region` | `enum` | north_america / europe / asia / latam / africa / south_america | Geographic market segment |

---

## 🚀 Setup & Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL *(or SQLite for quick local dev)*
- Gmail account with App Password enabled *(for OTP emails — optional, prints to console in dev mode)*

### 1. Clone

```bash
git clone https://github.com/krohitrkumar/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Python Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create `backend/.env`:

```env
# Database — postgresql:// for Postgres, sqlite:///./churn.db for local dev
DATABASE_URL=postgresql://user:password@localhost:5432/churn_db

# JWT
SECRET_KEY=your-256-bit-secret-here
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Gmail SMTP (optional — OTP prints to console if not set)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
SMTP_FROM_NAME=Customer Churn Intelligence
```

> **Gmail App Password:** Google Account → Security → 2-Step Verification → App passwords → generate for "Mail".

### 4. Train the ML Model

```bash
python ml_pipeline/train.py
```

Reads `ml_pipeline/data/processed.csv`, trains the Gradient Boosting pipeline, saves to `ml_pipeline/artifacts/churn_model.pkl`.

### 5. Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Interactive API docs available at `http://localhost:8000/docs`.

### 6. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite dev server proxies all `/api/*` requests to `http://localhost:8000`.

### 7. Create Your First Admin Account

Use the Swagger UI at `/docs` or call `POST /api/auth/register`:

```json
{
  "email": "admin@company.com",
  "password": "SecurePass123",
  "first_name": "Admin",
  "last_name": "User",
  "role": "admin"
}
```

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | SQLAlchemy-compatible DB URL |
| `SECRET_KEY` | ✅ | hardcoded fallback | JWT signing key — **change in production** |
| `ALGORITHM` | ❌ | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | ❌ | `1440` (24 h) | JWT token lifespan |
| `SMTP_HOST` | ❌ | `smtp.gmail.com` | SMTP server host |
| `SMTP_PORT` | ❌ | `587` | SMTP port |
| `SMTP_USER` | ❌ | — | Sender email — leave blank for dev console mode |
| `SMTP_PASSWORD` | ❌ | — | Gmail App Password |
| `SMTP_FROM_NAME` | ❌ | `Customer Churn Intelligence` | Display name in sent emails |
| `MODEL_PATH` | ❌ | auto-resolved | Absolute path to `churn_model.pkl` |

---

## 💡 Business Impact

| Capability | Business Value |
|---|---|
| Real-time churn probability | CSMs see exact risk scores before customer calls |
| Automated retention playbooks | Actions are pre-prescribed — no manual analysis needed |
| Bulk CSV/Excel import with ML scoring | Onboard 1000+ existing accounts in one upload with instant predictions |
| Role-based access control | Admins control data; CSMs operate; read-only users review |
| Risk breakdown by segment | Identify which contract types, regions, or payment methods have highest churn rates |
| Prediction history timeline | Track whether retention actions improved a customer's score over time |
| OTP passwordless login | Fast, secure access for field CSMs without password management overhead |
| Dev-mode OTP fallback | Zero config needed for local development — OTP prints to terminal |

---


MIT © [Rohit Kumar](https://github.com/krohitrkumar)

---

*Built with Python, FastAPI, React 18, Scikit-Learn, and SQLAlchemy.*
