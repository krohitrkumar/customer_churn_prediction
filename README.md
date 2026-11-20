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

| 🌐 Frontend (Vercel) | ⚡ Backend API (Render) | 📖 Swagger Docs |
|:--------------------:|:----------------------:|:---------------:|
| [customerchurnprediction-roan.vercel.app](https://customerchurnprediction-roan.vercel.app) | [retentrix-ai.onrender.com](https://retentrix-ai.onrender.com) | [retentrix-ai.onrender.com/docs](https://retentrix-ai.onrender.com/docs) |

</div>

---

## 📐 System Architecture

```
╔═══════════════════════════════════════════════════════════════╗
║           RETENTRIX PLATFORM — SYSTEM ARCHITECTURE           ║
╚═══════════════════════════════════════════════════════════════╝

                        👤 END USERS
              CSM  ·  Admin  ·  Analyst  ·  Viewer
                            │
                       Browser (HTTPS)
                            │
            ┌───────────────▼───────────────┐
            │   🌐 VERCEL  (Edge CDN)       │
            │  customerchurnprediction-     │
            │  roan.vercel.app              │
            │                               │
            │  /dashboard   /customers      │
            │  /predict     /roi-hub        │
            │  /settings    /verify-otp     │
            │  /login       /register       │
            └───────────────┬───────────────┘
                            │  REST / JSON (Axios)
            ┌───────────────▼───────────────┐
            │  ⚡ RENDER (Linux Container)  │
            │  retentrix-ai.onrender.com    │
            │                               │
            │  ┌─────────────────────────┐  │
            │  │    FastAPI  (main.py)   │  │
            │  │                         │  │
            │  │  /auth/*   Auth + OTP   │  │
            │  │  /customers/* CRUD +    │  │
            │  │             Import      │  │
            │  │  /predict/* ML + SHAP   │  │
            │  │             GenAI + RL  │  │
            │  │  /analytics/* KPI dash  │  │
            │  └─────────────────────────┘  │
            │                               │
            │  ┌─────────────────────────┐  │
            │  │    SERVICE  LAYER       │  │
            │  │                         │  │
            │  │  ml_service   → GBM     │  │
            │  │  shap_service → XAI     │  │
            │  │  genai_service→ LLM     │  │
            │  │  rl_service   → LinUCB  │  │
            │  │  email_service→ SMTP    │  │
            │  │  cache_service→ Redis   │  │
            │  └─────────────────────────┘  │
            └───────┬───────────────┬───────┘
                    │               │
        ┌───────────▼──┐    ┌───────▼────────┐
        │ 🐘 NEON DB   │    │ 🔴 UPSTASH     │
        │ PostgreSQL   │    │ Redis (TLS)    │
        │              │    │                │
        │ users        │    │ OTP codes      │
        │ customers    │    │ TTL 10 min     │
        │ predictions  │    │ Rate counters  │
        │ otp_verif.   │    └────────────────┘
        └──────────────┘
                            ┌────────────────┐
                            │ 📧 SMTP EMAIL  │
                            │ Gmail TLS/SSL  │
                            │                │
                            │ OTP delivery   │
                            │ Retention mail │
                            │ ADMIN+CSM only │
                            └────────────────┘

        ┌──────────────────────────────────────┐
        │  🤖 ML PIPELINE  (Offline Training)  │
        │                                      │
        │  data/customer_churn_dataset.csv     │
        │          │                           │
        │          ▼                           │
        │  notebooks/customer_churn.ipynb      │
        │  Feature Engineering + EDA           │
        │          │                           │
        │          ▼                           │
        │  GradientBoostingClassifier          │
        │  200 trees · depth 5 · lr 0.05       │
        │          │                           │
        │          ▼                           │
        │  artifacts/churn_model.pkl  ◄──────── loaded by ml_service.py
        └──────────────────────────────────────┘

        ┌──────────────────────────────────────┐
        │  🎯 REINFORCEMENT LEARNING ARMS      │
        │  Algorithm: LinUCB Contextual Bandit │
        │                                      │
        │  ARM 0 → Personalized Discount       │
        │  ARM 1 → Premium Upgrade Offer       │
        │  ARM 2 → Dedicated CSM Outreach      │
        │  ARM 3 → Loyalty Rewards Program     │
        │  ARM 4 → No Action (baseline)        │
        │                                      │
        │  Net ROI = LTV × P(retain|arm)       │
        │            − Intervention Cost       │
        └──────────────────────────────────────┘
```

---

## ✨ Enterprise Capabilities

### 1. 🔮 Explainable AI — SHAP TreeExplainer

Every prediction includes a SHAP force-plot decomposition. The model ranks each feature's contribution so your CSM team acts on evidence, not guesswork.

- **Algorithm:** `shap.TreeExplainer` on GradientBoostingClassifier
- **Output:** Per-customer top-N feature contributions with direction (↑ churn risk / ↓ churn risk)
- **API:** `POST /predict/single` → returns `shap_values`, `top_factors[]`

### 2. 💬 GenAI Retention Copilot

A conversational AI embedded in every customer detail page. It reads the full profile, churn probability, SHAP factors, and contract data — then generates bespoke retention strategies in natural language.

- **Model:** `qwen/qwen3-27b` via Groq inference
- **Context injected:** tenure, satisfaction, support calls, late payments, churn probability, top SHAP factors
- **API:** `POST /predict/genai-advice`

### 3. 🎯 Reinforcement Learning — LinUCB Bandit

Automatically selects the optimal retention intervention using **contextual multi-armed bandit (LinUCB)**. Balances exploration vs exploitation, updating arm rewards from real retention outcomes.

| Arm | Intervention | Typical Use Case |
|:---:|-------------|-----------------|
| 0 | Personalized Discount | Price-sensitive churners |
| 1 | Premium Upgrade Offer | Feature-gap churners |
| 2 | Dedicated CSM Outreach | Relationship-at-risk |
| 3 | Loyalty Rewards Program | Long-tenure dissatisfied |
| 4 | No Action | Low-churn-risk customers |

### 4. 📂 Resilient CSV / Excel Ingestion

The bulk import engine handles real-world dirty data exports from any CRM:

- **Alias mapping** — `tenure_months` = `Tenure Months` = `tenure months` — all resolved
- **Case-insensitive** — `Month-to-Month` = `month_to_month` = `MONTH TO MONTH`
- **Extra columns ignored** — Upload a 50-column dump; only the 7 ML features are extracted
- **Email optional** — If absent, email features are skipped gracefully
- **Formats** — `.xlsx` and `.csv` both supported
- **API:** `POST /customers/import`

### 5. 📊 Executive ROI Hub (`/roi-hub`)

| Widget | Insight |
|--------|---------|
| **LTV at Risk** | Total MRR × avg tenure of at-risk cohort |
| **ARR at Risk** | Annualized revenue exposed by churn probability |
| **Recoverable Revenue** | Revenue saveable with successful retention |
| **What-If Simulator** | Model retention rate vs revenue saved interactively |
| **Heavy Lifters** | High-LTV customers worth maximum investment |
| **High-Drain** | Customers where retention cost > LTV |
| **Methodology Card** | Full formula transparency for finance teams |

---

## 🔐 Authentication Flow

```
  REGISTER
  ────────
  Fill form → POST /auth/register
           → Account created (inactive)
           → OTP sent to email automatically
           → Enter OTP at /verify-otp
           → Account activated → JWT issued → Dashboard

  LOGIN
  ─────
  POST /auth/login → credentials verified → JWT (access token)

  OTP / FORGOT PASSWORD
  ──────────────────────
  POST /auth/send_otp → OTP dispatched → POST /auth/verify_otp
  → POST /auth/reset_password (new password, no old password needed)

  SECURITY LAYERS
  ───────────────
  • JWT HS256 tokens with configurable expiry
  • bcrypt password hashing (cost factor 12)
  • OTP stored in PostgreSQL with 10-minute TTL
  • Rate limiting: 30 logins/min, 20 OTPs/min per IP (SlowAPI)
  • RBAC roles: ADMIN · CSM · ANALYST (Viewer)
  • Retention email actions restricted to ADMIN + CSM only
```

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
| **ORM** | SQLAlchemy 2 | Database operations |
| **Validation** | Pydantic v2 | Request / response schemas |
| **Auth** | python-jose + bcrypt | JWT signing, password hashing |
| **ML Model** | scikit-learn GBM | Churn probability prediction |
| **Explainability** | SHAP TreeExplainer | Feature attribution |
| **GenAI** | Groq API (qwen3-27b) | Retention strategy copilot |
| **RL Bandit** | NumPy LinUCB | Intervention optimization |
| **Cache / OTP** | Redis (Upstash TLS) | OTP + rate limit storage |
| **Rate Limiting** | SlowAPI | Endpoint abuse prevention |
| **Email** | smtplib SMTP/TLS | OTP delivery, retention alerts |
| **Database** | PostgreSQL (Neon Cloud) | Persistent storage |
| **Frontend Host** | Vercel Edge CDN | Global frontend delivery |
| **Backend Host** | Render (Linux container) | Persistent SMTP-capable server |
| **ML Training** | Jupyter + scikit-learn | Offline model development |

---

## 🤖 ML Model Details

**Algorithm:** `GradientBoostingClassifier` (scikit-learn)

| Hyperparameter | Value |
|:--------------|:-----:|
| `n_estimators` | 200 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |

**Input Features (7 — derived from customer profile):**

| # | Feature | Type | Valid Values |
|:-:|---------|------|-------------|
| 1 | `tenure_months` | Integer | 0 – 100 |
| 2 | `support_calls` | Integer | 0 – 50 (ticket count) |
| 3 | `late_payments` | Integer | 0 – 30 |
| 4 | `satisfaction_score` | Float | 0.0 – 10.0 |
| 5 | `contract_type` | Categorical | `month_to_month` · `one_year` · `two_year` |
| 6 | `payment_method` | Categorical | `card` · `wallet` · `bank` |
| 7 | `region` | Categorical | `north_america` · `europe` · `asia` · `latam` · `africa` · `south_america` |

**Output:** Churn probability ∈ [0.0, 1.0] + risk label (`Low` / `Moderate` / `Critical`) + SHAP values

**Risk Thresholds:**

| Label | Probability Range |
|-------|:-----------------:|
| 🟢 Low | < 40% |
| 🟡 Moderate | 40% – 75% |
| 🔴 Critical | > 75% |

---

## 📊 ROI Hub — Economic Formulas

| Metric | Formula |
|--------|---------|
| **Customer LTV** | `Monthly Charges × Tenure Months` |
| **ARR at Risk** | `Monthly Charges × 12 × Churn Probability` |
| **Recoverable Revenue** | `ARR at Risk × Target Retention Rate` |
| **LinUCB Net ROI** | `LTV × P(retain \| arm) − Intervention Cost` |
| **Heavy Lifter** | `LTV > 75th percentile AND Churn Probability > 50%` |
| **High-Drain** | `Intervention Cost > Expected LTV Recovery` |

---

## 📁 Project Structure

```
customer_churn_prediction/
│
├── backend/                          FastAPI application
│   ├── main.py                       App init, CORS, router registration
│   ├── database/
│   │   ├── connection.py             SQLAlchemy engine + session
│   │   └── config.py                 Pydantic settings (reads .env)
│   ├── middlewares/
│   │   ├── auth_middleware.py        JWT decode, current_user injection
│   │   └── limiter.py                SlowAPI rate-limit instance
│   ├── models/
│   │   ├── user.py                   User + OTPVerification ORM models
│   │   ├── customer.py               Customer ORM model + enums
│   │   └── prediction.py             PredictionHistory ORM model
│   ├── schemas/
│   │   ├── auth.py                   Auth request/response schemas
│   │   ├── customer.py               Customer request/response schemas
│   │   └── prediction.py             Prediction schemas + PlaybookRec
│   ├── routes/
│   │   ├── auth.py                   /auth/* register, login, OTP, JWT
│   │   ├── customers.py              /customers/* CRUD, import, export
│   │   ├── predictions.py            /predict/* single, batch, GenAI, RL
│   │   └── analytics.py              /analytics/* dashboard KPIs
│   ├── services/
│   │   ├── ml_services.py            GBM inference + playbook engine
│   │   ├── shap_service.py           SHAP TreeExplainer, force plots
│   │   ├── genai_service.py          Groq LLM retention copilot
│   │   ├── rl_service.py             LinUCB 5-arm bandit
│   │   ├── email_service.py          SMTP OTP + retention alerts
│   │   └── cache_service.py          Redis TTL operations
│   ├── .env.example                  Environment variable template
│   └── .env                          (gitignored) live secrets
│
├── frontend/                         React + Vite SPA
│   └── src/
│       ├── api/                      Axios instance + endpoint helpers
│       ├── components/               Reusable UI components
│       ├── context/                  Auth context (JWT state)
│       ├── hooks/                    Custom React hooks
│       ├── pages/
│       │   ├── DashboardPage.jsx     KPI overview, churn metrics
│       │   ├── CustomersPage.jsx     Customer table, import/export
│       │   ├── CustomerDetailPage.jsx Profile + GenAI copilot
│       │   ├── PredictPage.jsx       Single and batch AI prediction
│       │   ├── RoiHubPage.jsx        Financial ROI dashboard
│       │   ├── SettingsPage.jsx      Profile, change password
│       │   ├── LoginPage.jsx         Login form
│       │   ├── RegisterPage.jsx      Registration + OTP trigger
│       │   └── OtpPage.jsx           OTP verification
│       ├── styles/                   Global CSS tokens
│       ├── App.jsx                   Route definitions
│       └── main.jsx                  Vite entry point
│
├── ml_pipeline/                      Offline ML training
│   ├── data/
│   │   ├── customer churn dataset.csv   Raw training data
│   │   └── processed.csv               Feature-engineered dataset
│   ├── notebooks/
│   │   └── customer_churn.ipynb     EDA + model training notebook
│   ├── artifacts/
│   │   └── churn_model.pkl          Trained GBM model (joblib)
│   ├── train.py                     CLI training script
│   └── requirements.txt
│
├── demo_customers_sample.csv         Sample upload file (CSV)
├── demo_customers_sample.xlsx        Sample upload file (Excel)
├── requirements.txt                  Backend production dependencies
├── Procfile                          Render start command
└── README.md
```

---

## ⚙️ Local Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL database — [Neon free tier](https://neon.tech)
- Redis instance — [Upstash free tier](https://upstash.com)
- Gmail account with **App Password** enabled
- Groq API key — [console.groq.com](https://console.groq.com) (free)

### 1 — Clone

```bash
git clone https://github.com/krohitrkumar/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2 — Backend

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Create `backend/.env`:

```env
DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname
REDIS_URL=rediss://default:token@host:port

SECRET_KEY=your-random-64-char-hex-string
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your-gmail-app-password
SMTP_FROM_NAME=Retentrix AI

GROQ_API_KEY=gsk_your_key_here
```

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API: `http://localhost:8000` | Swagger: `http://localhost:8000/docs`

### 3 — ML Model (retrain if needed)

```bash
cd ml_pipeline
pip install -r requirements.txt
python train.py
```

Saves trained model to `ml_pipeline/artifacts/churn_model.pkl`.

### 4 — Frontend

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

Frontend: `http://localhost:5173`

---

## 🚀 Render Deployment Guide

### Backend (Web Service)

| Setting | Value |
|---------|-------|
| **Environment** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |

**Environment Variables** (add in Render dashboard):

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | `postgresql+asyncpg://...` (Neon connection string) |
| `REDIS_URL` | `rediss://...` (Upstash connection string) |
| `SECRET_KEY` | 64-char random hex string |
| `ALGORITHM` | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` |
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | `your@gmail.com` |
| `SMTP_PASSWORD` | Gmail App Password |
| `SMTP_FROM_NAME` | `Retentrix AI` |
| `GROQ_API_KEY` | `gsk_...` |

### Frontend (Vercel)

| Setting | Value |
|---------|-------|
| **Framework Preset** | Vite |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Env Variable** | `VITE_API_BASE_URL=https://retentrix-ai.onrender.com` |

---

## 🎓 Academic Context

This platform was developed as a **final-year capstone project** at:

**Dr. Akhilesh Das Gupta Institute of Professional Studies (ADGIPS)**
*Affiliated to Guru Gobind Singh Indraprastha University (GGSIPU)*
B.Tech — Artificial Intelligence & Data Science

The project demonstrates full end-to-end applied AI:

| Domain | Implementation |
|--------|---------------|
| Supervised Learning | Gradient Boosting churn classifier |
| Explainable AI | SHAP TreeExplainer force plots |
| Reinforcement Learning | LinUCB contextual multi-armed bandit |
| Generative AI | LLM-powered retention strategy copilot |
| Production Deployment | Render + Vercel + Neon + Upstash |
| Enterprise UX | RBAC, OTP auth, ROI dashboards |

---

## 👨‍💻 Author

**Rohit Kumar**
- 🐙 GitHub: [@krohitrkumar](https://github.com/krohitrkumar)
- 📧 Email: rohitkumar.adgitm.aids@gmail.com
- 🏫 ADGIPS, GGSIPU — B.Tech AI & Data Science

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

**Built with ❤️ for AI-first retention teams**

*Retentrix — Don't lose customers. Understand them.*

</div>
