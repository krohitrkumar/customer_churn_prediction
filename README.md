# 🧠 Retentrix — Customer Churn Intelligence Platform

> An enterprise-grade, full-stack Machine Learning web application that predicts customer churn in real-time, scores account risk levels, manages customer portfolios, and delivers automated retention playbooks — powered by a serialized Gradient Boosting pipeline, a FastAPI REST backend, and a modern React 18 frontend built through AI prompt engineering.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://customerchurnprediction-roan.vercel.app)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)](https://customer-churn-prediction-backend-y0q2.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon.tech-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://neon.tech)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 🚀 Live Deployments

| Component | Platform | URL |
| :--- | :--- | :--- |
| **Frontend Web App** | Vercel | [https://customerchurnprediction-roan.vercel.app](https://customerchurnprediction-roan.vercel.app) |
| **Backend REST API** | Render | [https://customer-churn-prediction-backend-y0q2.onrender.com](https://customer-churn-prediction-backend-y0q2.onrender.com) |
| **Interactive API Docs** | Swagger UI | [https://customer-churn-prediction-backend-y0q2.onrender.com/docs](https://customer-churn-prediction-backend-y0q2.onrender.com/docs) |
| **Database** | Neon PostgreSQL | Cloud Serverless PostgreSQL (`ep-winter-lab...aws.neon.tech`) |

---

## 📑 Table of Contents

1. [Overview & Key Features](#-overview--key-features)
2. [AI-Driven Frontend Engineering](#-ai-driven-frontend-engineering-prompt-engineering)
3. [System Architecture](#-system-architecture)
4. [Machine Learning Pipeline](#-machine-learning-pipeline)
5. [Backend Architecture (FastAPI & SQLAlchemy)](#-backend-architecture-fastapi--sqlalchemy)
   - [Database Schema (Neon PostgreSQL)](#database-schema-neon-postgresql)
   - [API Reference](#api-reference)
   - [Authentication & Role-Based Access Control](#authentication--role-based-access-control-rbac)
   - [Services & Performance](#services--performance)
6. [Frontend UI & Experience](#-frontend-ui--experience)
7. [Dataset & Feature Schema](#-dataset--feature-schema)
8. [Local Development & Setup](#-local-development--setup)
9. [Production Deployment Guide](#-production-deployment-guide)
10. [Business Impact & Playbooks](#-business-impact--playbooks)

---

## 🎯 Overview & Key Features

Retentrix provides Customer Success Managers (CSMs), Growth Teams, and Enterprise Executives with an actionable **churn intelligence command center**:

- 🧠 **Real-Time ML Churn Scoring:** Ingests behavioral, contract, and billing metrics to output exact probability percentages (0–100%) and categorizes risk into **Critical**, **Moderate**, and **Low**.
- 📋 **Automated Retention Playbooks:** Generates prescriptive, rule-driven playbook actions customized to why each customer is at risk (e.g. support queue prioritization, contract discounts, onboarding reviews).
- 📂 **Smart Bulk CSV & Excel Uploader:** Client-side column pre-validation, drag-and-drop file ingestion, automatic duplicate skipping, and batch ML inference for thousands of accounts.
- 📊 **Executive Analytics & Segment Breakdown:** Dynamic risk distribution across contract lengths, regions (LATAM, North America, Europe, Asia Pacific), and payment methods.
- 🔐 **Enterprise Security & RBAC:** Passwords hashed with bcrypt, stateless JWT tokens with 24-hour expiration, passwordless 6-digit email OTP login, and role-based permissions (`admin`, `csm`, `user`).
- ⚡ **1-Click Seed Generator:** Generates 50 realistic demo enterprise accounts with instant AI predictions for immediate evaluation.

---

## 🤖 AI-Driven Frontend Engineering (Prompt Engineering)

The entire frontend client was architected, styled, and implemented leveraging **advanced AI pair-programming and prompt engineering**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROMPT ENGINEERING METHODOLOGY                       │
│                                                                         │
│  Domain & Requirements   ──►  Iterative AI Generation  ──►   Component  │
│  Specification                • Radix UI Primitives          Integration│
│  • Token Normalization        • SVG Visualizations           & Cache    │
│  • RBAC UI Gates              • Responsive Breakpoints       Validation │
└─────────────────────────────────────────────────────────────────────────┘
```

### Highlights of the AI-Engineered Frontend:
* **Zero Frontend Boilerplate Knowledge Barrier:** Architected from high-level behavioral prompts into a modular, production-ready React 18 Single-Page Application.
* **Component-Driven Design System:** Glassmorphic UI, custom CSS variables, accessible Radix UI dialogs, select dropdowns, and toast notification systems.
* **Interactive Data Visualizations:** Custom SVG-based animated Churn Gauges and dynamic donut charts with hover glow effects and segment breakdowns.
* **Client-Side Data Resilience:** Module-level caching with 30-second stale-while-revalidate TTL in custom hooks (`useCustomers.js`) to eliminate navigation lag and unnecessary network re-fetches.
* **Smart File Upload Pipeline:** Client-side CSV/Excel header inspector validating required schema before payload transfer, paired with dedicated 5-minute timeout clients for large datasets.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       CLIENT TIER (React 18 + Vite)                     │
│                                                                         │
│   Dashboard  │  Customer CRM  │  ML Predictor  │  CSV/Excel Importer    │
│   • Hosted on Vercel Edge Network                                       │
│   • Global Axios Client with JWT Auto-Injection                         │
│   • Client-Side Pre-Validation & In-Memory Shared Cache                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTPS / REST (JSON)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       BACKEND API TIER (FastAPI)                        │
│                                                                         │
│  ┌───────────────────────┐  ┌──────────────────────┐  ┌──────────────┐  │
│  │   Auth & Security     │  │   Customer Engine    │  │  Analytics   │  │
│  │   • JWT / HS256       │  │   • CRUD Operations  │  │  • Aggregate │  │
│  │   • bcrypt Hashing    │  │   • Batch Ingestion  │  │  • Breakdowns│  │
│  │   • RBAC Middleware   │  │   • 1-Click Seeder   │  │  • Top At Risk│ │
│  └───────────┬───────────┘  └──────────┬───────────┘  └───────┬──────┘  │
│              │                         │                      │         │
│  ┌───────────▼─────────────────────────▼──────────────────────▼──────┐  │
│  │                       Services Layer                              │  │
│  │   • MLService: Gradient Boosting Pipeline Inference (Scikit-Learn)│  │
│  │   • EmailService: SMTP HTML OTP Delivery Engine                   │  │
│  │   • ProcessTimeMiddleware: Request Latency Header Tracking        │  │
│  └─────────────────────────────────────┬─────────────────────────────┘  │
└────────────────────────────────────────┼────────────────────────────────┘
                                         │ SQLAlchemy 2.0 ORM
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DATABASE & STORAGE TIER                           │
│                                                                         │
│   Neon Serverless PostgreSQL (Production) / SQLite (Local Fallback)     │
│   • `users` (Credentials & Roles)                                       │
│   • `customers` (Account Attributes & Latest Churn Scores)              │
│   • `prediction_history` (Audit Log of Historical Predictions)          │
│   • `otp_verifications` (Time-bound OTP Tokens)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Machine Learning Pipeline

### 1. Model Architecture
The underlying champion model is a serialized **Scikit-Learn Pipeline** consisting of two integrated stages:

```
Raw Customer Features (7 parameters)
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1: ColumnTransformer Preprocessing                                │
│                                                                         │
│ • Categorical Features: OneHotEncoder(drop='first', handle_unknown='ignore')│
│   - `contract_type` (month_to_month, one_year, two_year)                │
│   - `payment_method` (card, wallet, bank)                               │
│   - `region` (north_america, europe, asia, latam, africa, south_america)│
│                                                                         │
│ • Numerical Features: Passthrough                                       │
│   - `tenure_months`, `support_calls`, `late_payments`, `satisfaction_score`│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Transformed Feature Vector
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 2: GradientBoostingClassifier                                     │
│                                                                         │
│ • n_estimators = 200        • learning_rate = 0.05                      │
│ • max_depth = 5             • min_samples_split = 5                     │
│ • min_samples_leaf = 2      • random_state = 42                         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
                      predict_proba[:, 1] × 100
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
   Churn Probability Score                     Risk Band Classification
          (0.0% – 100.0%)                      Critical / Moderate / Low
```

### 2. Risk Classification Rules

| Probability Score | Risk Band | Color Tag | System Implication |
| :--- | :--- | :--- | :--- |
| **> 75.0%** | 🔴 **Critical** | `#EF4444` | High likelihood of cancellation; immediate action required |
| **40.0% – 75.0%** | 🟠 **Moderate** | `#F59E0B` | Showing early friction signals; proactive outreach needed |
| **< 40.0%** | 🟢 **Low** | `#10B981` | Healthy, stable engagement; standard maintenance |

### 3. Automated Retention Playbooks

The backend evaluates customer profiles against 5 condition triggers to output personalized retention plays:

| Trigger Condition | Playbook Category | Prescribed Intervention |
| :--- | :--- | :--- |
| `satisfaction_score <= 3.0` | **Customer Satisfaction** | Trigger immediate executive outreach and dispatch a diagnostic feedback survey. |
| `support_calls >= 5` | **Support Queue Priority** | Assign a Senior Technical Specialist to resolve outstanding friction tickets. |
| `late_payments >= 3` | **Billing Flexibility** | Offer payment restructuring or automated installment reminder workflow. |
| `tenure_months < 12` | **Onboarding Retention** | Enroll customer in high-touch onboarding check-in call program. |
| `contract_type == month_to_month` | **Contract Commitment** | Offer a 15% discount incentive for upgrading to an annual subscription plan. |
| *(None of above)* | **Healthy Account** | Account is stable. Maintain standard automated quarterly product updates. |

---

## ⚙️ Backend Architecture (FastAPI & SQLAlchemy)

### Database Schema (Neon PostgreSQL)

```sql
-- Users & Access Control
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user' NOT NULL, -- 'admin', 'csm', 'user'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Customer Database
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    customer_code VARCHAR(50) UNIQUE NOT NULL,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    tenure_months INTEGER NOT NULL,
    support_calls INTEGER DEFAULT 0,
    late_payments INTEGER DEFAULT 0,
    satisfaction_score FLOAT NOT NULL,
    contract_type VARCHAR(50) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    region VARCHAR(50) NOT NULL,
    latest_churn_score FLOAT,
    latest_risk_level VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE
);

-- Prediction Audit History
CREATE TABLE prediction_history (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id) ON DELETE CASCADE,
    triggered_by_user_id INTEGER REFERENCES users(id),
    churn_probability FLOAT NOT NULL,
    risk_level VARCHAR(50) NOT NULL,
    playbook_recommendations JSON,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6-Digit Email OTP Store
CREATE TABLE otp_verifications (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    otp_code VARCHAR(6) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

### API Reference

All routes are mounted under the `/api` prefix. Interactive documentation is available at `/docs`.

#### 🔐 Authentication (`/api/auth`)
| Method | Endpoint | Auth | Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/auth/register` | ❌ | Any | Register user profile & password |
| `POST` | `/auth/login` | ❌ | Any | OAuth2 form login (`username`, `password`) → Returns JWT |
| `GET` | `/auth/me` | ✅ | Any | Retrieve authenticated profile |
| `POST` | `/auth/send_otp` | ❌ | Any | Send 6-digit email OTP (10-minute validity) |
| `POST` | `/auth/verify_otp` | ❌ | Any | Verify OTP → Returns access token |
| `POST` | `/auth/change_password` | ✅ | Any | Authenticated password change (validates old password) |
| `POST` | `/auth/reset_password` | ❌ | Any | Reset password via verified OTP |

#### 👥 Customer Management (`/api/customers`)
| Method | Endpoint | Auth | Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/customers/` | ✅ | Any | Paginated customer listing (`skip`, `limit`) |
| `POST` | `/customers/` | ✅ | `admin`, `csm` | Register a single customer account |
| `GET` | `/customers/{id}` | ✅ | Any | Fetch customer profile by ID |
| `PUT` | `/customers/{id}` | ✅ | `admin`, `csm` | Update customer parameters |
| `DELETE` | `/customers/{id}` | ✅ | `admin` | Permanently delete customer record |
| `POST` | `/customers/seed` | ✅ | `admin`, `csm` | Seed 50 demo accounts with live ML scores |
| `POST` | `/customers/upload_file` | ✅ | `admin`, `csm` | Bulk import CSV/Excel spreadsheets with ML scoring |

#### 🧠 Predictions (`/api/predict`)
| Method | Endpoint | Auth | Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/predict/single` | ✅ | Any | Run real-time churn prediction on profile |
| `GET` | `/predict/history/{customer_id}` | ✅ | Any | Retrieve chronological prediction history for customer |

#### 📊 Executive Analytics (`/api/analytics`)
| Method | Endpoint | Auth | Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/analytics/summary` | ✅ | Any | Top-level KPIs (total accounts, avg churn, risk counts) |
| `GET` | `/analytics/risk_breakdown` | ✅ | Any | Segment aggregations (by contract, region, payment) |
| `GET` | `/analytics/top_at_risk` | ✅ | Any | Top critical accounts ordered by risk score |
| `GET` | `/analytics/recent_activity` | ✅ | Any | Real-time stream of latest predictions |

---

### Authentication & Role-Based Access Control (RBAC)

* **Hashing:** `bcrypt` with unique salt generation per credential.
* **Token Transport:** RFC 6750 `Authorization: Bearer <JWT_TOKEN>`.
* **Roles:**
  * `admin`: Complete administrative control, customer deletion, seed generation, user management.
  * `csm`: Customer creation, profile modification, bulk upload, prediction execution.
  * `user`: Read-only access to customer data, individual prediction tests, dashboard metrics.

---

## 💻 Dataset & Feature Schema

The Gradient Boosting pipeline expects 7 features:

| Feature | Type | Valid Range | Business Meaning & Significance |
| :--- | :--- | :--- | :--- |
| **`tenure_months`** | Integer | `1 – 72` | Months of continuous service. Higher tenure strongly reduces churn risk. |
| **`support_calls`** | Integer | `0 – 20` | Support tickets opened. High volume indicates unresolved product friction. |
| **`late_payments`** | Integer | `0 – 12` | Invoices paid after due date. Direct leading indicator of payment churn. |
| **`satisfaction_score`** | Float | `1.0 – 10.0` | Customer CSAT rating. Ratings `<= 3.0` trigger critical escalation. |
| **`contract_type`** | Categorical | `month_to_month`, `one_year`, `two_year` | Subscription term length. Month-to-month contracts have highest volatility. |
| **`payment_method`** | Categorical | `card`, `wallet`, `bank` | Primary billing method. |
| **`region`** | Categorical | `north_america`, `europe`, `asia`, `latam`, `africa`, `south_america` | Geographic market segment. |

---

## 🛠️ Local Development & Setup

### Prerequisites
* **Python 3.10+**
* **Node.js 18+** & **npm**
* **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/krohitrkumar/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Backend Setup
```bash
# Create and activate Python virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your settings:
```env
DATABASE_URL=sqlite:///./churn_database.db
SECRET_KEY=your_development_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

Start the FastAPI development server:
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Swagger UI will be live at `http://localhost:8000/docs`.

### 3. Frontend Setup
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
The client dashboard will be available at `http://localhost:3000` (or `http://localhost:5173`), with all `/api` requests proxied to the backend automatically.

### 4. (Optional) Retrain ML Model Pipeline
```bash
python ml_pipeline/train.py
```
This reads `ml_pipeline/data/processed.csv`, evaluates model recall/accuracy, and exports `ml_pipeline/artifacts/churn_model.pkl`.

---

## 🌐 Production Deployment Guide

### Deploy Backend to Render

1. Create a new **Web Service** on [Render](https://dashboard.render.com).
2. Connect your GitHub repository: `krohitrkumar/customer_churn_prediction`.
3. Configure settings:
   * **Root Directory:** `backend`
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r ../requirements.txt`
   * **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variables in the Render Dashboard:
   * `PYTHON_VERSION` = `3.11.9`
   * `DATABASE_URL` = `postgresql://<user>:<password>@<host>/<database>?sslmode=require`
   * `SECRET_KEY` = `<your-secure-random-key>`
   * `SMTP_USER` = `<your-gmail-address>` *(optional for OTP)*
   * `SMTP_PASSWORD` = `<your-gmail-app-password>` *(optional for OTP)*

---

### Deploy Frontend to Vercel

1. Import your GitHub repository on [Vercel](https://vercel.com/dashboard).
2. Configure project settings:
   * **Root Directory:** `frontend`
   * **Framework Preset:** `Vite`
   * **Build Command:** `npm run build`
   * **Output Directory:** `dist`
3. Add Environment Variable:
   * `VITE_API_BASE_URL` = `https://<your-render-backend-url>/api`
4. Click **Deploy**.

---

## 📈 Business Impact & Playbooks

| Operational Metric | Before Retentrix | With Retentrix Platform |
| :--- | :--- | :--- |
| **Churn Identification** | Reactive (after cancellation notice) | **Proactive (30–60 days before contract expiry)** |
| **CSM Triage Efficiency** | Manual spreadsheet aggregation | **Automated risk prioritization (Top-At-Risk Feed)** |
| **Action Prescriptions** | Ad-hoc discounts | **Context-aware retention playbooks** |
| **Batch Account Onboarding**| Slow manual data entry | **1-click CSV/Excel batch ML prediction import** |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Architected & Built by **[Rohit Kumar](https://github.com/krohitrkumar)**.*
