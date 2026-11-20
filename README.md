<div align="center">

<img src="assets/logo.svg" width="72" height="72" alt="Retentrix Logo" />

# Retentrix
### AI Customer Churn Intelligence & Retention Copilot

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon_Cloud-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://neon.tech)
[![Redis](https://img.shields.io/badge/Redis-Upstash_Cloud-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://upstash.com)
[![Render](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://render.com)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/krohitrkumar)

<br/>

> **Enterprise churn intelligence and revenue protection platform** purpose-built for B2B SaaS teams. Combines Explainable AI (SHAP TreeExplainer), Generative AI Retention Copilot (Groq / Qwen 3 27B), Reinforcement Learning (LinUCB Contextual Bandit), and an executive financial ROI Hub.

<br/>

| 🌐 Live Frontend (Vercel) | ⚡ Live Backend API (Render) | 📖 Interactive API Docs (Swagger) |
|:-------------------------:|:---------------------------:|:----------------------------------:|
| [customerchurnprediction-roan.vercel.app](https://customerchurnprediction-roan.vercel.app) | [retentrix-ai.onrender.com](https://retentrix-ai.onrender.com) | [retentrix-ai.onrender.com/docs](https://retentrix-ai.onrender.com/docs) |

</div>

---

## 📐 System Architecture & Data Flow

<div align="center">
  <a href="assets/architecture.png" target="_blank">
    <img src="assets/architecture.svg" alt="Retentrix End-to-End System Architecture" width="100%" />
  </a>
  <p align="center">
    <sub>🔍 <i>Click on the architecture diagram above or <a href="assets/architecture.png" target="_blank">open the 4K High-Resolution PNG</a> to zoom in on any module, service, or data flow connector.</i></sub>
  </p>
</div>

---

## 🔄 End-to-End Architectural Flow

The Retentrix platform operates as a cohesive, closed-loop machine learning and customer retention ecosystem:

```
[Raw Dataset] ──► [Jupyter Notebook] ──► [churn_model.pkl]
                                                 │
                                                 ▼
[Frontend Users] ◄──► [React SPA] ◄──► [FastAPI API] ◄──► [AI Services] ◄──► [Neon DB / Redis]
 (CSM / Admin)        (Vercel)           (Render)          (SHAP/Groq/RL)       (Cloud Storage)
```

### 1. Offline Machine Learning Lifecycle (`ml_pipeline/`)
1. **Raw Ingestion**: Historical behavioral telemetry is staged at `ml_pipeline/data/customer churn dataset.csv`, containing 7 core behavioral features (`tenure_months`, `support_calls`, `late_payments`, `satisfaction_score`, `contract_type`, `payment_method`, `region`).
2. **Exploratory Data Analysis & Tuning**: The `notebooks/customer_churn.ipynb` notebook performs outlier treatment, categorical one-hot encoding, feature normalization, and stratified cross-validation.
3. **Model Selection**: Multiple classifiers (Logistic Regression, Random Forest, XGBoost, and Gradient Boosting) are benchmarked. `GradientBoostingClassifier` is selected based on superior F1 score and calibrated probability distributions.
4. **Artifact Export**: The model is serialized into `ml_pipeline/artifacts/churn_model.pkl` via `joblib.dump()`.

### 2. Backend Services & AI Inferencing (`backend/services/`)
At server startup on **Render**, FastAPI initializes the core service layer:
- **`ml_services.py`**: Loads `churn_model.pkl` directly into RAM. Exposes `predict_churn()` for real-time scoring and `predict_churn_batch()` for vector-accelerated NumPy inference. Categorizes accounts into **Low (<40%)**, **Moderate (40–75%)**, and **Critical (>75%)** risk tiers, and triggers automated retention playbooks.
- **`shap_service.py`**: Initializes a `shap.TreeExplainer` over the gradient booster to calculate local Shapley attributions, isolating the top positive and negative behavioral drivers for every customer.
- **`genai_service.py`**: Connects to the Groq Cloud LPU inference API running `qwen/qwen3-27b`. Ingests customer metrics, contractual parameters, and SHAP drivers to synthesize personalized, natural-language outreach scripts for Customer Success Managers (CSMs).
- **`rl_service.py`**: Formulates retention optimization as a Contextual Multi-Armed Bandit using **LinUCB** across 5 distinct action arms (Discounts, Upgrades, Dedicated Outreach, Loyalty Rewards, Baseline No Action). Selects the intervention that maximizes Net Return on Investment (ROI).
- **`email_service.py`**: Dispatches inline-styled HTML one-time passcodes (OTP) and role-restricted CSM retention outreach via secure SMTP/TLS (port 587).
- **`cache_service.py`**: Interfaces with Upstash Cloud Redis for token-bucket rate limiting (SlowAPI) and transient OTP verification states.

### 3. API Routing & Controller Layer (`backend/routes/`)
All client interactions flow through FastAPI endpoints running inside a persistent Linux container on Render:
- **`/auth/*`**: Handles registration with an automatic email OTP verification gate (`is_active = False` until verified), OAuth2 password-flow login returning signed HS256 JWT tokens, and role-based access control (RBAC).
- **`/customers/*`**: Powers customer CRUD, export, and schema-tolerant CSV/Excel bulk ingestion with intelligent column alias resolution.
- **`/predict/*`**: Coordinates single-record scoring, batch scoring, SHAP explainability matrices, GenAI copilot recommendations, and LinUCB arm selection.
- **`/analytics/*`**: Aggregates macro KPIs, global churn percentage, risk distributions, and Annual Recurring Revenue (ARR) exposed to churn.

### 4. Managed Cloud Infrastructure
- **Neon Cloud PostgreSQL**: Serverless PostgreSQL database hosting four relational entities: `users` (credentials, roles, activation status), `customers` (behavioral features and latest churn metrics), `predictions` (audit logs and feature snapshots), and `otp_verifications` (passcodes with 10-minute TTL).
- **Upstash Cloud Redis**: Low-latency in-memory data store for distributed IP rate limiting (20 OTP/min, 30 login/min) and session caching.

### 5. Client Application Layer (`frontend/`)
Deployed on **Vercel's Global Edge CDN**:
- Built with **React 18** and **Vite 5**, utilizing modern CSS custom properties and Radix UI primitives.
- An Axios client instance (`src/api/client.js`) automatically binds `Authorization: Bearer <JWT>` tokens to outgoing requests and handles automatic 401 unauthenticated session resets.
- Provides 6 operational views: Executive Dashboard, Customers Management, Customer Detail with GenAI Copilot, AI Real-Time Predictor, Executive ROI Hub, and Security Settings.

---

## ✨ Core Enterprise Features

### 1. 🔮 Explainable AI (SHAP TreeExplainer)
Retentrix replaces "black-box" predictions with mathematically sound feature attribution. Every prediction computes local Shapley values:
- Identifies exact behavioral drivers (e.g. `+3.27 Risk Driver: Support Calls` vs `-1.05 Protective: Satisfaction Score`).
- Equips CSMs with factual evidence before contacting customer stakeholders.
- Endpoint: `POST /predict/single` returns `shap_values` and ranked `top_factors[]`.

### 2. 💬 GenAI Retention Copilot
Embedded conversational AI that transforms raw machine learning data into actionable customer success strategies:
- **Inference Engine**: Groq Cloud LPU running `qwen/qwen3-27b`.
- **Dynamic Context Injection**: Customer tenure, MRR, contract type, satisfaction score, churn probability, and top SHAP factors are injected into the prompt context.
- **Output**: Bespoke outreach strategies, negotiation levers, and custom email drafts.
- Endpoint: `POST /predict/genai-advice`.

### 3. 🎯 Reinforcement Learning (LinUCB Contextual Bandit)
Rather than applying generic discounts to all churners, Retentrix prescribes optimal interventions:
- Evaluates 5 operational action arms:
  - **Arm 0**: Personalized Discount Incentive
  - **Arm 1**: Premium Feature Upgrade
  - **Arm 2**: Dedicated Executive CSM Outreach
  - **Arm 3**: Loyalty Rewards Program
  - **Arm 4**: No Action (Baseline)
- Balances exploration and exploitation via upper confidence bounds, optimizing the financial objective:
$$\text{Net ROI} = \text{LTV} \times P(\text{retain} \mid \text{arm}) - \text{Intervention Cost}$$

### 4. 📂 Resilient CSV & Excel Ingestion
Designed for real-world enterprise CRM exports (Salesforce, HubSpot, Stripe):
- **Intelligent Alias Resolution**: Automatically maps variations such as `Tenure Months`, `tenure_months`, `tenure months`, or `tenuremonths`.
- **Case-Insensitive Normalization**: Resolves variations like `month-to-month`, `Month to Month`, and `MONTH_TO_MONTH`.
- **Schema-Tolerant**: Safely processes files with extraneous columns, extracting only the 7 model features.
- **Dual Format Support**: Accepts both `.csv` and `.xlsx` workbooks.
- Endpoint: `POST /customers/import`.

### 5. 📊 Executive ROI Hub (`/roi-hub`)
CFO-grade financial dashboard translating predictive probabilities into monetary impact:
- **LTV at Risk**: Cumulative lifetime value of accounts in Critical or Moderate risk tiers.
- **ARR at Risk**: Annualized recurring revenue exposed to imminent churn.
- **Recoverable ARR**: Potential revenue saved through active retention workflows.
- **What-If Simulator**: Interactive slider modeling retention rate scenarios against recovered cash flow.
- **Cohort Segregation**: Automatically isolates **Heavy Lifters** (high LTV accounts worth defending) from **High-Drain** accounts (where intervention cost exceeds LTV).

---

## 🛠️ Technology Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Frontend Framework** | React 18 + Vite 5 | Single-page application rendering and state management |
| **Routing** | React Router DOM v6 | Client-side view transitions and route guards |
| **HTTP Client** | Axios | REST communication with JWT interceptors |
| **UI Primitives** | Radix UI + CSS Variables | Accessible dialogs, selects, and design tokens |
| **Auth State** | React Context API + jwt-decode | JWT session lifecycle management |
| **Backend Framework** | FastAPI 0.115 (Python 3.11) | High-performance asynchronous REST API |
| **Database ORM** | SQLAlchemy 2.0 (asyncpg) | Asynchronous database operations |
| **Data Validation** | Pydantic v2 | Schema definition and strict request validation |
| **Security & Auth** | python-jose + bcrypt | HS256 JWT generation and password hashing |
| **Predictive Model** | scikit-learn GBM | GradientBoostingClassifier churn scoring |
| **Explainability** | SHAP | TreeExplainer feature attribution |
| **Generative AI** | Groq Cloud (Qwen 3 27B) | Real-time retention playbook generation |
| **Bandit Optimizer** | NumPy (LinUCB) | Contextual multi-armed bandit optimization |
| **Relational Database** | Neon Cloud PostgreSQL | Managed serverless database |
| **In-Memory Cache** | Upstash Cloud Redis | Distributed rate limiting and OTP storage |
| **Rate Limiter** | SlowAPI | Endpoint DDoS and brute-force mitigation |
| **Email Transport** | smtplib (STARTTLS / SSL) | OTP and retention notification delivery |
| **Frontend Hosting** | Vercel Edge Network | Global CDN static asset delivery |
| **Backend Hosting** | Render Linux Container | Persistent containerized web service |

---

## 🤖 Machine Learning Specifications

**Algorithm**: `GradientBoostingClassifier` (`sklearn.ensemble`)

| Hyperparameter | Value | Description |
|:--------------|:-----:|:------------|
| `n_estimators` | 200 | Number of sequential boosting stages |
| `max_depth` | 5 | Maximum tree depth per weak learner |
| `learning_rate` | 0.05 | Shrinkage step size per iteration |
| `subsample` | 0.8 | Stochastic sample fraction per tree |

### Model Input Features (7)

| Feature Name | Type | Valid Range / Categories | Description |
|:------------|:----:|:------------------------|:------------|
| `tenure_months` | Integer | `0` to `100` | Account tenure with the company |
| `support_calls` | Integer | `0` to `50` | Support tickets logged |
| `late_payments` | Integer | `0` to `30` | Incurred late payment instances |
| `satisfaction_score` | Float | `0.0` to `10.0` | Customer satisfaction rating |
| `contract_type` | Categorical | `month_to_month`, `one_year`, `two_year` | Active commitment structure |
| `payment_method` | Categorical | `card`, `wallet`, `bank` | Primary payment instrument |
| `region` | Categorical | `north_america`, `europe`, `asia`, `latam`, `africa`, `south_america` | Geographic operating region |

### Risk Classification Thresholds

| Risk Level | Probability Range | Default Automated Playbook |
|:----------:|:-----------------:|:--------------------------|
| 🟢 **Low** | `p < 0.40` | Account healthy. Standard quarterly automated updates. |
| 🟡 **Moderate** | `0.40 ≤ p ≤ 0.75` | Proactive CSM check-in and feature adoption audit. |
| 🔴 **Critical** | `p > 0.75` | Immediate executive outreach and tailored LinUCB intervention. |

---

## 🔐 Authentication & Security Architecture

```
  REGISTRATION FLOW (Gated by OTP)
  ────────────────────────────────
  1. POST /auth/register ──► Creates user with is_active = False
  2. System generates secure 6-digit OTP (10-minute expiry)
  3. SMTP dispatches inline HTML verification code
  4. User enters code at /verify-otp
  5. POST /auth/verify_otp ──► Validates OTP, activates user (is_active = True), issues JWT
  
  LOGIN FLOW
  ──────────
  POST /auth/login (OAuth2 Password Request Form)
  ──► Verifies bcrypt hash (Cost factor 12)
  ──► Checks is_active == True (Rejects unverified accounts)
  ──► Returns HS256 Bearer JWT Access Token
  
  PASSWORD RECOVERY FLOW
  ──────────────────────
  POST /auth/send_otp ──► Generates fresh OTP
  POST /auth/reset_password ──► Updates password via verified OTP (no old password required)
  
  ROLE-BASED ACCESS CONTROL (RBAC)
  ────────────────────────────────
  • ADMIN:   Full system privileges, deletion, role assignment, customer management
  • CSM:     Customer management, running AI predictions, sending retention emails
  • USER:    Read-only analytics and customer browsing
```

---

## 📁 Repository Directory Structure

```
customer_churn_prediction/
│
├── assets/                           Visual documentation assets
│   ├── logo.svg                      Official Retentrix vector logo
│   ├── architecture.svg              Vector system architecture diagram
│   └── architecture.png              4K high-resolution architecture export
│
├── backend/                          FastAPI production application
│   ├── main.py                       FastAPI initialization, CORS regex, route mounts
│   ├── database/
│   │   ├── config.py                 Pydantic application settings (.env parser)
│   │   ├── connection.py             Async SQLAlchemy engine & session factory
│   │   └── security.py               JWT token encoding/decoding & bcrypt hashing
│   ├── middlewares/
│   │   ├── auth_middleware.py        FastAPI JWT dependency injection
│   │   └── limiter.py                SlowAPI rate limiting instance
│   ├── models/                       SQLAlchemy ORM models
│   │   ├── user.py                   User (with roles & is_active) & OTPVerification
│   │   ├── customer.py               Customer record & categorical enums
│   │   └── prediction.py             PredictionHistory audit logs
│   ├── schemas/                      Pydantic v2 request & response schemas
│   │   ├── auth.py                   Auth, OTP, and password schemas
│   │   ├── customer.py               Customer input, update & response models
│   │   └── prediction.py             Prediction, SHAP, GenAI & RL schemas
│   ├── routes/                       REST API endpoint controllers
│   │   ├── auth.py                   /auth/* (register, login, OTP, reset)
│   │   ├── customers.py              /customers/* (CRUD, import, export)
│   │   ├── predictions.py            /predict/* (single, batch, SHAP, GenAI, RL)
│   │   └── analytics.py              /analytics/* (KPIs, churn rates, ARR at risk)
│   ├── services/                     Domain business logic & AI engines
│   │   ├── ml_services.py            GBM inference & automated playbooks
│   │   ├── shap_service.py           SHAP TreeExplainer local attribution
│   │   ├── genai_service.py          Groq LLM retention copilot integration
│   │   ├── rl_service.py             LinUCB 5-arm contextual bandit
│   │   ├── email_service.py          SMTP/TLS OTP & retention mail dispatch
│   │   └── cache_service.py          Redis client operations
│   ├── .env.example                  Backend environment variable template
│   └── .env                          Local environment secrets (gitignored)
│
├── frontend/                         React 18 + Vite client SPA
│   ├── src/
│   │   ├── api/                      Axios client instance & service endpoints
│   │   ├── components/               Modular UI components (tables, gauges, buttons)
│   │   ├── context/                  AuthContext providing JWT session state
│   │   ├── pages/                    Application views
│   │   │   ├── DashboardPage.jsx     Executive overview, churn metrics & charts
│   │   │   ├── CustomersPage.jsx     Customer directory & bulk file uploader
│   │   │   ├── CustomerDetailPage.jsx Profile, SHAP feature plot & GenAI Copilot
│   │   │   ├── PredictPage.jsx       Interactive single & batch predictor
│   │   │   ├── RoiHubPage.jsx        Financial ROI Hub & What-If simulator
│   │   │   ├── SettingsPage.jsx      User profile & password management
│   │   │   ├── LoginPage.jsx         OAuth2 credentials login
│   │   │   ├── RegisterPage.jsx      Account signup with automated OTP dispatch
│   │   │   └── OtpPage.jsx           6-digit security code verification view
│   │   ├── styles/                   Global CSS design tokens and layout styling
│   │   ├── App.jsx                   Route declarations and protection gates
│   │   └── main.jsx                  React DOM application mount point
│   ├── .env                          Local frontend configuration
│   └── .env.example
│
├── ml_pipeline/                      Offline model research & training
│   ├── data/
│   │   ├── customer churn dataset.csv   Raw training data (7 features + label)
│   │   └── processed.csv               Engineered and encoded dataset
│   ├── notebooks/
│   │   ├── customer_churn.ipynb      Comprehensive EDA, modeling & evaluation
│   │   └── requirements.txt          Jupyter notebook dependencies
│   ├── artifacts/
│   │   └── churn_model.pkl           Trained GradientBoostingClassifier artifact
│   ├── train.py                      CLI script to retrain and serialize model
│   └── requirements.txt
│
├── demo_customers_sample.csv         Sample test file for CSV import
├── demo_customers_sample.xlsx        Sample test file for Excel import
├── requirements.txt                  Backend production dependencies
├── Procfile                          Render deployment start command
└── README.md
```

---

## ⚙️ Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL database ([Neon Serverless Postgres](https://neon.tech))
- Redis database ([Upstash Serverless Redis](https://upstash.com))
- Gmail account with an **App Password** (for SMTP OTP delivery)
- Groq Cloud API key ([console.groq.com](https://console.groq.com))

### 1. Repository Setup
```bash
git clone https://github.com/krohitrkumar/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

Create `backend/.env` with your credentials:
```env
DATABASE_URL=postgresql+asyncpg://user:password@host/dbname
REDIS_URL=rediss://default:token@host:port

SECRET_KEY=generate_a_random_64_character_hex_string
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
SMTP_FROM_NAME=Retentrix AI

GROQ_API_KEY=gsk_your_groq_api_key
```

Run the backend development server:
```bash
cd backend
uvicorn main:app --reload --port 8000
```
- API Base URL: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`

### 3. Frontend Setup
```bash
cd frontend
npm install
```

Create `frontend/.env`:
```env
VITE_API_BASE_URL=http://localhost:8000
```

Start the Vite development server:
```bash
npm run dev
```
- Frontend Application: `http://localhost:5173`

---

## 🚀 Cloud Deployment Specifications

### Backend (Render Web Service)
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**: Configure `DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`, `ALGORITHM`, `SMTP_*`, and `GROQ_API_KEY` in the Render dashboard.

### Frontend (Vercel)
- **Framework Preset**: Vite
- **Root Directory**: `frontend`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Environment Variables**: `VITE_API_BASE_URL=https://retentrix-ai.onrender.com`

---

## 🎓 Academic Capstone Context

This platform was designed and developed as a **final-year capstone project** at:

**Dr. Akhilesh Das Gupta Institute of Professional Studies (ADGIPS)**  
*Affiliated with Guru Gobind Singh Indraprastha University (GGSIPU), New Delhi*  
**Department of Artificial Intelligence & Data Science**

Demonstrates production-grade integration across five core computer science and AI disciplines:
1. **Supervised Machine Learning**: Calibrated Gradient Boosting churn classification.
2. **Explainable AI (XAI)**: Local feature attribution via game-theoretic Shapley values.
3. **Reinforcement Learning**: LinUCB contextual bandit for decision optimization.
4. **Applied Generative AI**: Context-grounded LLM reasoning for customer success.
5. **Modern Cloud Systems Engineering**: Scalable decoupled architecture on Render, Vercel, Neon, and Upstash.

---

## 👨‍💻 Author

**Rohit Kumar**
- 🐙 **GitHub**: [@krohitrkumar](https://github.com/krohitrkumar)
- 💼 **LinkedIn**: [Rohit Kumar](https://www.linkedin.com/in/krohitrkumar)
- 📧 **Email**: rohitkumar.adgitm.aids@gmail.com
- 🏫 **Institution**: ADGIPS, GGSIPU — B.Tech Artificial Intelligence & Data Science

*Solo Project developed by Rohit Kumar with AI assistance. Proprietary Academic & Capstone Work.*

---

<div align="center">

*Retentrix — Don't lose customers. Understand them.*

</div>
