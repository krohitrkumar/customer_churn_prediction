# 🧠 Retentrix: AI Customer Churn Intelligence & Retention Copilot

**Retentrix** is an enterprise-grade AI customer churn prediction and retention platform. It goes beyond basic probability classification by combining **Machine Learning (Gradient Boosting)**, **Explainable AI (SHAP TreeExplainer)**, **Generative AI (Groq Cloud LLM)**, and **Reinforcement Learning (LinUCB Contextual Multi-Armed Bandit)** to predict customer attrition, pinpoint exact risk drivers, simulate portfolio Net ROI, and prescribe automated retention playbooks.

---

## 🏗️ High-Level System Architecture

```
                               ┌─────────────────────────────────────────────────────┐
                               │                 React 18 + Vite SPA                 │
                               │  • Real-time Churn Probability & SVG Risk Gauge    │
                               │  • Resilient CSV & Excel File Ingestion Engine      │
                               │  • SHAP Impact & Feature Attribution Visualizer     │
                               │  • Retentrix AI Copilot with 1-Click Direct Email   │
                               │  • Executive ROI & Revenue Hub (/roi-hub)           │
                               └──────────────────────────┬──────────────────────────┘
                                                          │
                                                          │ HTTPS / JSON (REST API)
                                                          ▼
                               ┌─────────────────────────────────────────────────────┐
                               │           FastAPI Backend (Render Linux VM)         │
                               │  • JWT Authentication & Role-Based Access Control   │
                               │  • Asynchronous Rate Limiting (SlowAPI)             │
                               │  • Vectorized Batch Processing (NumPy & Pandas)     │
                               │  • Vectorized Multi-Threaded SMTP Dispatch          │
                               └───────┬──────────┬──────────┬──────────┬──────────┬─┘
                                       │          │          │          │          │
         ┌─────────────────────────────┘          │          │          │          └─────────────────────────────┐
         ▼                                        ▼          │          ▼                                        ▼
┌───────────────────────┐ ┌────────────────────────┐         │ ┌────────────────────────┐ ┌────────────────────────┐
│  Supervised ML Engine │ │ Explainable AI (SHAP)  │         │ │ Retentrix GenAI Agent  │ │ Reinforcement Learning │
│  • GradientBoosting   │ │ • TreeExplainer Model  │         │ │ • Groq Cloud Engine    │ │ • LinUCB Contextual    │
│  • 7 Behavioral Inputs│ │ • Feature Attributions │         │ │ • qwen/qwen3.8-27b     │ │   Multi-Armed Bandit   │
│  • Calibrated Prob.   │ │ • Risk vs Protective   │         │ │ • Root-Cause Diagnosis │ │ • Action Cost vs Net   │
│    (0% to 100%)       │ │   Factor Analysis      │         │ │ • Direct Email Drafting│ │   LTV ROI Optimization │
└───────────────────────┘ └────────────────────────┘         │ └────────────────────────┘ └────────────────────────┘
                                                             │
                                                             ▼
                                 ┌─────────────────────────────────────────────────────┐
                                 │                Cloud Infrastructure                 │
                                 │  • Neon Cloud: Serverless PostgreSQL Database       │
                                 │  • Upstash Cloud: Redis Caching Layer (<5ms)        │
                                 │  • Gmail SMTP: Port 465 SSL / 587 TLS Verification  │
                                 │  • Render: Continuous Web Service Container         │
                                 └─────────────────────────────────────────────────────┘
```

---

## ✨ Enterprise Capabilities

### 1. Explainable AI (SHAP TreeExplainer)
Rather than acting as an uninterpretable black box, Retentrix computes Shapley values for each individual prediction:
* Identifies the primary **Risk Drivers** pushing the customer toward churn (e.g., month-to-month contracts, high technical support tickets, low CSAT rating).
* Highlights **Protective Factors** mitigating risk (e.g., long subscription tenure, automatic billing).
* Quantifies exact percentage impact for every behavioral attribute.

### 2. Generative AI Retention Copilot
Powered by Groq Cloud (`qwen/qwen3.8-27b`):
* **Root-Cause Diagnosis:** Synthesizes SHAP attributions into executive bullet points explaining why the customer is dissatisfied.
* **Prescriptive Action Plan:** Formulates step-by-step strategies for Customer Success Managers (CSMs).
* **Direct Outreach Email Generation:** Dynamically drafts personalized retention emails signed by the logged-in CSM.
* **1-Click SMTP Dispatch:** Role-guarded email dispatch with dynamic `From` and `Reply-To` headers.

### 3. Reinforcement Learning (LinUCB Contextual Multi-Armed Bandit)
* Implements a 5-arm Contextual Bandit:
  1. `Standard Proactive Nudge` ($0 cost)
  2. `Concierge Onboarding Review` ($25 cost)
  3. `15% Annual Renewal Discount` ($50 cost)
  4. `Priority Feature Tier Upgrade` ($75 cost)
  5. `Executive Sponsor Intervention` ($120 cost)
* Optimizes for **Net ROI**: $\text{Net ROI} = \mathbb{E}[\text{Saved Revenue}] - \text{Intervention Cost}$.
* Dynamically incorporates real-time outcome feedback (+Reward / -Reward) to balance exploration and exploitation.

### 4. Resilient CSV & Excel Ingestion
* Supports both `.csv` and `.xlsx` files up to 5,000 rows.
* Intelligent column alias mapping (e.g., `Customer ID`, `customer_code`, `Tenure Months`, `tenure_months`, `CSAT`, `satisfaction`).
* Normalized contract pricing tolerance for Netflix-style plans, quarterly billing, and custom commitments.
* Vectorized batch prediction (`predict_churn_batch`) and batch database insertion (`db.add_all()`).

### 5. Executive Revenue & ROI Hub (`/roi-hub`)
* Real-time portfolio financial health tracking.
* **Interactive What-If Churn Reduction Simulator:** Models how reducing churn by 2% to 30% affects annual recurring revenue and net profit.
* **Heavy Lifters (VIP Accounts):** Highlights high-LTV customers requiring immediate retention attention.
* **High-Drain Accounts:** Flags accounts where service and incentive costs exceed recoverable lifetime value.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 18, Vite, Radix UI Primitives, Axios, Custom Modular CSS |
| **Backend** | Python 3.11, FastAPI, SQLAlchemy 2.0 ORM, Pydantic v2, Uvicorn |
| **Machine Learning** | Scikit-Learn 1.7.1, GradientBoostingClassifier, Joblib, NumPy, Pandas |
| **Explainable AI** | SHAP (TreeExplainer) |
| **Generative AI** | Groq Cloud API (`qwen/qwen3.8-27b`) |
| **Reinforcement Learning** | LinUCB Contextual Multi-Armed Bandit (NumPy) |
| **Caching** | Upstash Redis Cloud |
| **Database** | Neon Serverless PostgreSQL (Production) / SQLite (Local Development) |
| **Email Service** | Gmail SMTP SSL (Port 465) / TLS (Port 587) |
| **Deployment** | Render (Backend Web Service Linux Container) & Vercel (Frontend SPA) |

---

## 🚀 Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/krohitrkumar/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Configure Backend
```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in `backend/.env`:
```env
DATABASE_URL=postgresql://<user>:<password>@<neon-host>/neondb?sslmode=require
SECRET_KEY=your-secure-jwt-secret-key
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=qwen/qwen3.8-27b
REDIS_URL=rediss://default:<token>@<upstash-endpoint>:6379
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-gmail-16-char-app-password
SMTP_FROM_NAME=Retentrix Intelligence
```

Start the FastAPI development server:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs will be available at `http://localhost:8000/docs`.

### 3. Configure Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## ☁️ Production Deployment on Render

### Backend Web Service Settings
1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your GitHub repository: `https://github.com/krohitrkumar/customer_churn_prediction`.
3. Configure settings:
   - **Environment:** `Python`
   - **Region:** `Singapore` or `Oregon`
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variables:
   - `DATABASE_URL`
   - `SECRET_KEY`
   - `GROQ_API_KEY`
   - `GROQ_MODEL` (`qwen/qwen3.8-27b`)
   - `REDIS_URL`
   - `SMTP_USER`
   - `SMTP_PASSWORD`
   - `SMTP_FROM_NAME`

---

## 👨‍💻 Author

**Rohit Kumar**
* GitHub: [@krohitrkumar](https://github.com/krohitrkumar)
* Contact: [https://github.com/k_rohit_r_kumar](https://github.com/k_rohit_r_kumar)
