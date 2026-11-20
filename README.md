<div align="center">

# 🧠 Retentrix
### AI Customer Churn Intelligence & Retention Copilot

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://neon.tech)
[![Redis](https://img.shields.io/badge/Redis-Upstash-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://upstash.com)
[![Render](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://render.com)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> **Enterprise-grade churn intelligence platform** combining Explainable AI (SHAP), Generative AI retention copilot, Reinforcement Learning (LinUCB), and an executive ROI Hub — built for B2B SaaS teams.

<br/>

| 🌐 Frontend | ⚡ Backend API | 📖 Swagger Docs |
|:-----------:|:--------------:|:---------------:|
| [customerchurnprediction-roan.vercel.app](https://customerchurnprediction-roan.vercel.app) | [retentrix-ai.onrender.com](https://retentrix-ai.onrender.com) | [/docs](https://retentrix-ai.onrender.com/docs) |

</div>

---

## 📐 System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           RETENTRIX PLATFORM ARCHITECTURE                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              👤  END USERS                                      │
│          CSM / Admin / Analyst  ──  Browser (React SPA on Vercel)               │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │  HTTPS
┌──────────────────────────────────────▼──────────────────────────────────────────┐
│                        🌐  VERCEL  (Frontend CDN / Edge)                        │
│   customerchurnprediction-roan.vercel.app                                       │
│                                                                                 │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│   │ Dashboard  │  │ Customers  │  │ AI Predict │  │  ROI Hub   │              │
│   │  /dash     │  │ /customers │  │  /predict  │  │  /roi-hub  │              │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘              │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐                               │
│   │  Settings  │  │  OTP Page  │  │  Auth      │                               │
│   │  /settings │  │  /verify   │  │  /login    │                               │
│   └────────────┘  └────────────┘  └────────────┘                               │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │  REST / JSON  (Axios)
┌──────────────────────────────────────▼──────────────────────────────────────────┐
│                     ⚡  RENDER  (Backend — Persistent Linux Container)           │
│   retentrix-ai.onrender.com                                                     │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                       FastAPI Application (main.py)                     │   │
│   │                                                                         │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │   │
│   │  │  auth.py     │  │customers.py  │  │predictions.py│  │analytics.py│ │   │
│   │  │  /auth/*     │  │/customers/*  │  │/predict/*    │  │/analytics/*│ │   │
│   │  │              │  │              │  │              │  │            │ │   │
│   │  │ • Register   │  │ • List/CRUD  │  │ • Predict    │  │• Dashboard │ │   │
│   │  │ • Login      │  │ • CSV/Excel  │  │ • Batch      │  │• KPI stats │ │   │
│   │  │ • OTP send   │  │   Import     │  │ • SHAP xai   │  │• Churn rate│ │   │
│   │  │ • OTP verify │  │ • Export     │  │ • GenAI chat │  │            │ │   │
│   │  │ • JWT tokens │  │ • Alias map  │  │ • RL action  │  │            │ │   │
│   │  │ • RBAC       │  │              │  │              │  │            │ │   │
│   │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                            SERVICE LAYER                                │   │
│   │                                                                         │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │   │
│   │  │ ml_service   │  │ shap_service │  │genai_service │  │ rl_service │ │   │
│   │  │              │  │              │  │              │  │            │ │   │
│   │  │ GBM Predict  │  │ TreeExplain  │  │ Groq LLM     │  │  LinUCB   │ │   │
│   │  │ 7 features   │  │ Force plots  │  │ qwen3-27b    │  │  5 arms   │ │   │
│   │  │ Prob score   │  │ Top factors  │  │ Copilot chat │  │  Net ROI  │ │   │
│   │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │   │
│   │                                                                         │   │
│   │  ┌──────────────┐  ┌──────────────┐                                    │   │
│   │  │email_service │  │cache_service │                                    │   │
│   │  │              │  │              │                                    │   │
│   │  │ SMTP/TLS OTP │  │ Redis TTL    │                                    │   │
│   │  │ Inline HTML  │  │ OTP store    │                                    │   │
│   │  │ RBAC guards  │  │ Rate tokens  │                                    │   │
│   │  └──────────────┘  └──────────────┘                                    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
└───────────┬─────────────────────────────────┬───────────────────────────────────┘
            │  SQLAlchemy / asyncpg            │  redis-py / TLS
┌───────────▼──────────────┐      ┌───────────▼──────────────────────────────────┐
│  🐘  NEON CLOUD           │      │  🔴  UPSTASH CLOUD REDIS                     │
│  PostgreSQL (production)  │      │                                              │
│                           │      │  • OTP codes (TTL 5 min)                     │
│  Project: Churn Prediction│      │  • Rate limit counters                       │
│  Branch:  production      │      │  • Session caching                           │
│  ID: broad-surf-80458890  │      │                                              │
│                           │      │  TLS encrypted, globally distributed         │
│  Tables:                  │      └──────────────────────────────────────────────┘
│  • users                  │
│  • customers              │      ┌──────────────────────────────────────────────┐
│  • predictions            │      │  📧  SMTP EMAIL SERVICE (Gmail / Any SMTP)   │
│  • audit_logs             │      │                                              │
└───────────────────────────┘      │  • Registration OTP  (HTML inline styled)    │
                                   │  • Password reset OTP                        │
                                   │  • Retention alerts  (ADMIN / CSM only)      │
                                   │  • Rate limited: 20 OTP / min               │
                                   └──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        🤖  ML PIPELINE  (Offline Training)                      │
│   ml_pipeline/                                                                  │
│                                                                                 │
│   customer churn dataset.csv                                                    │
│          │                                                                      │
│          ▼                                                                      │
│   customer_churn.ipynb  ──► Feature Engineering ──► GradientBoostingClassifier │
│                                                      (200 trees, depth 5)       │
│                                                             │                   │
│                                                             ▼                   │
│                                                    artifacts/churn_model.pkl    │
│                                                    (loaded by ml_service.py)    │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌────────────────────────────────────┐
                    │   🔄  REINFORCEMENT LEARNING ARMS   │
                    │                                     │
                    │  ARM 0 ──► Personalized Discount    │
                    │  ARM 1 ──► Premium Upgrade Offer    │
                    │  ARM 2 ──► Dedicated CSM Outreach   │
                    │  ARM 3 ──► Loyalty Rewards Program  │
                    │  ARM 4 ──► No Action (baseline)     │
                    │                                     │
                    │  Algorithm : LinUCB (contextual)    │
                    │  Reward    : Net ROI per customer   │
                    │  Formula   : LTV × P(retain|arm)   │
                    │             − Intervention Cost     │
                    └────────────────────────────────────┘
```

---

## ✨ Enterprise Capabilities

### 1. 🔮 Explainable AI — SHAP TreeExplainer

Every prediction comes with a **SHAP force plot** decomposition. The model doesn't just say "this customer will churn" — it explains *why*, ranking the top contributing behavioral signals so your CSM team can act on evidence, not guesswork.

- **Algorithm:** `shap.TreeExplainer` on GradientBoostingClassifier
- **Output:** Per-customer top-5 feature contributions with direction (↑ churn / ↓ churn)
- **API:** `POST /predict/single` → returns `shap_values`, `top_factors[]`

### 2. 💬 GenAI Retention Copilot

A conversational AI agent embedded in every customer detail page. It reads the customer's full profile, churn probability, SHAP factors, and contract data — then generates bespoke retention strategies in natural language.

- **Model:** `qwen/qwen3-27b` via Groq inference API
- **Context injected:** tenure, MRR, satisfaction score, churn probability, top SHAP factors
- **API:** `POST /predict/genai-advice` → streaming retention recommendation

### 3. 🎯 Reinforcement Learning — LinUCB Bandit

Automatically selects the optimal retention intervention arm per customer using **contextual multi-armed bandit** (LinUCB). The algorithm balances exploration vs exploitation, updating arm rewards from actual retention outcomes.

| Arm | Intervention | Typical Use Case |
|-----|-------------|-----------------|
| 0 | Personalized Discount | Price-sensitive churners |
| 1 | Premium Upgrade Offer | Feature-gap churners |
| 2 | Dedicated CSM Outreach | Relationship-at-risk |
| 3 | Loyalty Rewards Program | Long-tenure dissatisfied |
| 4 | No Action (baseline) | Low-churn-risk customers |

**Net ROI Formula:**
```
Net ROI = LTV × P(retain | arm) − Intervention Cost
```

### 4. 📂 Resilient CSV / Excel Ingestion

The bulk import engine is **schema-tolerant by design** — it handles real-world dirty data from any SaaS export (Salesforce, HubSpot, Mixpanel, etc.):

- **Alias mapping:** `tenure_months` = `Tenure Months` = `tenure months` = `tenuremonths` — all resolved
- **Case normalization:** `Month-to-Month` = `month-to-month` = `MONTH TO MONTH`
- **Extra columns ignored:** Upload a 50-column CRM dump; only the 7 ML features are used
- **Email column optional:** If absent, email-based features are simply skipped
- **Excel + CSV:** Both `.xlsx` and `.csv` accepted
- **API:** `POST /customers/import`

### 5. 📊 Executive ROI Hub (`/roi-hub`)

A CFO-ready dashboard surfacing the financial reality of your churn risk:

| Widget | Insight |
|--------|---------|
| **LTV at Risk** | Total MRR × avg tenure of at-risk cohort |
| **ARR at Risk** | Annualized revenue exposed by churn probability |
| **Recoverable Revenue** | Revenue saveable with successful retention |
| **What-If Simulator** | Model retention rate vs revenue saved |
| **Heavy Lifters** | High-LTV customers worth maximum investment |
| **High-Drain** | Customers where retention cost > LTV (let go) |
| **Methodology Card** | Full formula transparency for finance teams |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + Vite 5 | SPA, component rendering |
| **Routing** | React Router DOM v6 | Client-side navigation |
| **HTTP Client** | Axios | API calls with interceptors |
| **UI Primitives** | Radix UI | Accessible dialogs, selects |
| **Auth State** | Context API + jwt-decode | JWT session management |
| **Backend** | FastAPI 0.115 | Async REST API, OpenAPI docs |
| **ORM** | SQLAlchemy 2 (async) | Database operations |
| **Validation** | Pydantic v2 | Request/response schemas |
| **Auth** | python-jose + bcrypt | JWT signing, password hashing |
| **ML Model** | scikit-learn GBM | Churn probability prediction |
| **Explainability** | SHAP TreeExplainer | Feature attribution |
| **GenAI** | Groq API (qwen3-27b) | Retention strategy generation |
| **RL Bandit** | NumPy LinUCB | Intervention optimization |
| **Cache / OTP** | Redis (Upstash) | TTL-based OTP storage |
| **Rate Limiting** | SlowAPI | Endpoint abuse prevention |
| **Email** | smtplib + SMTP TLS | OTP delivery, alerts |
| **Database** | PostgreSQL (Neon) | Persistent storage |
| **Frontend Host** | Vercel (Edge CDN) | Global frontend delivery |
| **Backend Host** | Render (Linux container) | Persistent SMTP-capable server |
| **ML Training** | Jupyter + scikit-learn | Offline model development |

---

## 🤖 ML Model Details

**Algorithm:** `GradientBoostingClassifier` (scikit-learn)

| Hyperparameter | Value |
|---------------|-------|
| `n_estimators` | 200 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |

**Input Features (7):**

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `tenure_months` | Integer | Months as a customer |
| 2 | `monthly_charges` | Float | Current monthly spend (\$) |
| 3 | `num_products` | Integer | Number of products subscribed |
| 4 | `satisfaction_score` | Float | CSAT score (1–5) |
| 5 | `support_tickets` | Integer | Open / recent support tickets |
| 6 | `contract_type` | Categorical | Month-to-Month / One Year / Two Year |
| 7 | `payment_method` | Categorical | Electronic / Mailed Check / Bank Transfer / Credit Card |

**Output:** Churn probability ∈ [0.0, 1.0] + binary label + SHAP values

---

## 🔐 Authentication & Security

```
Registration Flow:
  User fills form → POST /auth/register → OTP sent to email
       → POST /auth/verify-otp → Account activated → JWT issued

Login Flow:
  POST /auth/login → credentials verified → JWT (access + refresh tokens)

Password Reset:
  POST /auth/send-otp → OTP to email → POST /auth/verify-otp
       → POST /auth/change-password (with current password)

Security Layers:
  • JWT (HS256) with configurable expiry
  • bcrypt password hashing (cost factor 12)
  • Redis OTP store with 5-minute TTL
  • Rate limiting: 30 logins/min, 20 OTPs/min per IP
  • RBAC: roles = [ADMIN, CSM, ANALYST]
  • Email actions (send alerts) restricted to ADMIN + CSM
```

---

## 📁 Project Structure

```
customer_churn_prediction/
│
├── backend/                        # FastAPI application
│   ├── main.py                     # App init, CORS, router registration
│   ├── database/
│   │   └── connection.py           # Async SQLAlchemy engine + session
│   ├── middlewares/
│   │   └── auth_middleware.py      # JWT decode, current_user injection
│   ├── models/                     # SQLAlchemy ORM models
│   │   ├── user.py
│   │   ├── customer.py
│   │   └── prediction.py
│   ├── schemas/                    # Pydantic v2 request/response schemas
│   │   ├── auth.py
│   │   ├── customer.py
│   │   └── prediction.py
│   ├── routes/                     # API route handlers
│   │   ├── auth.py                 # /auth/* — register, login, OTP, JWT
│   │   ├── customers.py            # /customers/* — CRUD, import, export
│   │   ├── predictions.py          # /predict/* — single, batch, GenAI, RL
│   │   └── analytics.py            # /analytics/* — dashboard KPIs
│   ├── services/                   # Business logic layer
│   │   ├── ml_services.py          # GBM model inference
│   │   ├── shap_service.py         # SHAP TreeExplainer, force plots
│   │   ├── genai_service.py        # Groq LLM retention copilot
│   │   ├── rl_service.py           # LinUCB 5-arm bandit
│   │   ├── email_service.py        # SMTP OTP + retention alerts
│   │   └── cache_service.py        # Redis TTL operations
│   ├── .env.example                # Environment variable template
│   └── .env                        # (gitignored) live secrets
│
├── frontend/                       # React + Vite SPA
│   ├── src/
│   │   ├── api/                    # Axios instance + endpoint helpers
│   │   ├── components/             # Reusable UI components
│   │   ├── context/                # Auth context (JWT state)
│   │   ├── hooks/                  # Custom React hooks
│   │   ├── pages/
│   │   │   ├── DashboardPage.jsx   # KPI overview, churn rate chart
│   │   │   ├── CustomersPage.jsx   # Customer table, import/export
│   │   │   ├── CustomerDetailPage.jsx  # Profile + GenAI copilot
│   │   │   ├── PredictPage.jsx     # Single & batch AI prediction
│   │   │   ├── RoiHubPage.jsx      # Financial ROI dashboard
│   │   │   ├── SettingsPage.jsx    # Profile, change password
│   │   │   ├── LoginPage.jsx       # Login form
│   │   │   ├── RegisterPage.jsx    # Registration + OTP trigger
│   │   │   └── OtpPage.jsx         # OTP verification
│   │   ├── styles/                 # Global CSS tokens
│   │   ├── App.jsx                 # Route definitions
│   │   └── main.jsx                # Vite entry point
│   ├── .env                        # VITE_API_BASE_URL=https://retentrix-ai.onrender.com
│   └── .env.example
│
├── ml_pipeline/                    # Offline ML training
│   ├── data/
│   │   ├── customer churn dataset.csv   # Raw training data
│   │   └── processed.csv               # Feature-engineered data
│   ├── notebooks/
│   │   ├── customer_churn.ipynb    # EDA + model training notebook
│   │   └── requirements.txt        # Notebook-specific deps
│   ├── artifacts/
│   │   └── churn_model.pkl         # Trained GBM model (joblib)
│   ├── train.py                    # CLI training script
│   └── requirements.txt
│
├── demo_customers_sample.csv       # Sample upload file for demos
├── demo_customers_sample.xlsx      # Sample upload file (Excel)
├── requirements.txt                # Backend production deps
├── Procfile                        # Render start command
├── main.py                         # Root entry (imports backend)
└── README.md
```

---

## ⚙️ Local Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A PostgreSQL database (Neon free tier works)
- A Redis instance (Upstash free tier works)
- A Gmail account with App Password enabled (for SMTP)
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository

```bash
git clone https://github.com/krohitrkumar/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Backend Setup

```bash
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

Create `backend/.env`:

```env
DATABASE_URL=postgresql+asyncpg://user:password@host/dbname
REDIS_URL=rediss://default:token@host:port
SECRET_KEY=your-jwt-secret-key-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-gmail-app-password
FROM_EMAIL=your-email@gmail.com
FROM_NAME=Retentrix AI

GROQ_API_KEY=gsk_your_groq_key_here
```

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API live at: `http://localhost:8000` | Swagger: `http://localhost:8000/docs`

### 3. ML Model (if retraining)

```bash
cd ml_pipeline
pip install -r requirements.txt
python train.py
```

The trained model saves to `ml_pipeline/artifacts/churn_model.pkl`.

### 4. Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

```bash
npm run dev
```

Frontend live at: `http://localhost:5173`

---

## 🚀 Render Deployment Guide

### Backend (Web Service)

| Setting | Value |
|---------|-------|
| **Environment** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free (or Starter for no sleep) |

**Environment Variables** (add all from `.env` above in Render dashboard):

```
DATABASE_URL          postgresql+asyncpg://...
REDIS_URL             rediss://...
SECRET_KEY            <random 64-char string>
ALGORITHM             HS256
ACCESS_TOKEN_EXPIRE_MINUTES   60
REFRESH_TOKEN_EXPIRE_DAYS     7
SMTP_HOST             smtp.gmail.com
SMTP_PORT             587
SMTP_USER             your@gmail.com
SMTP_PASSWORD         your-app-password
FROM_EMAIL            your@gmail.com
FROM_NAME             Retentrix AI
GROQ_API_KEY          gsk_...
```

### Frontend (Vercel)

| Setting | Value |
|---------|-------|
| **Framework Preset** | Vite |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Environment Variable** | `VITE_API_BASE_URL=https://retentrix-ai.onrender.com` |

---

## 📊 ROI Hub — Economic Formulas

The ROI Hub uses the following financial models:

| Metric | Formula |
|--------|---------|
| **Customer LTV** | `Monthly Charges × Tenure Months` |
| **ARR at Risk** | `Monthly Charges × 12 × Churn Probability` |
| **Recoverable Revenue** | `ARR at Risk × Target Retention Rate` |
| **LinUCB Net ROI** | `LTV × P(retain \| arm) − Intervention Cost` |
| **Heavy Lifter Score** | `LTV > P75 AND Churn Probability > 0.5` |
| **High-Drain Score** | `Intervention Cost > Expected LTV Recovery` |

---

## 🔄 Data Flow

```
CSV / Excel Upload
       │
       ▼
  Alias Resolver ──► Normalize column names (case, spaces, synonyms)
       │
       ▼
  Schema Validator ──► Check required 7 features, fill defaults
       │
       ▼
  Bulk Insert ──► PostgreSQL customers table
       │
       ▼
  GBM Inference ──► churn_probability ∈ [0, 1]
       │
       ├──► SHAP TreeExplainer ──► top_factors[] (why this score?)
       │
       ├──► LinUCB Bandit ──► recommended_arm (which intervention?)
       │
       └──► Groq LLM ──► retention_strategy (what to say/do?)
                │
                ▼
         Dashboard KPIs ──► ROI Hub ──► Executive Report
```

---

## 🎓 Academic Context

This project was developed as a **final-year capstone project** at **ADGIT Delhi (AI/DS program)**. It demonstrates end-to-end applied AI:

- Supervised learning (churn classification)
- Explainable AI (SHAP)
- Reinforcement learning (LinUCB bandit)
- Generative AI (LLM-powered copilot)
- Production deployment (Render + Vercel + Neon + Upstash)
- Enterprise SaaS UX (RBAC, OTP auth, ROI dashboards)

---

## 👨‍💻 Author

**Rohit Kumar**
- 🐙 GitHub: [@k_rohit_r_kumar](https://github.com/k_rohit_r_kumar)
- 📧 Email: rohitkumar.adgitm.aids@gmail.com
- 🏫 ADGIT Delhi — B.Tech AI & Data Science

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for AI-first retention teams**

*Retentrix — Don't lose customers. Understand them.*

</div>
