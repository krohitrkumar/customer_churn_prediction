# 🧠 Customer Churn Prediction & Retention Intelligence System

Hey there! 👋 Welcome to my **Customer Churn Prediction & Retention Intelligence System**. 

This is a complete full-stack web application that uses Machine Learning to predict whether a customer is likely to leave (churn) or stay. It doesn’t just give a probability score — it also automatically generates **smart retention playbooks** (action plans) tailored to why the customer might be unhappy (like high support calls, low satisfaction rating, or late payment history).

---

## 🔗 Live Working Project

You can try the live application directly in your browser:

* 🌐 **Frontend Web App (Live on Vercel):** [https://customerchurnprediction-roan.vercel.app](https://customerchurnprediction-roan.vercel.app)
* ⚡ **Production Backend API (Live on Railway):** [https://web-production-f4a4f.up.railway.app](https://web-production-f4a4f.up.railway.app)
* 📖 **Interactive Swagger API Docs:** [https://web-production-f4a4f.up.railway.app/docs](https://web-production-f4a4f.up.railway.app/docs)
* 🐘 **Database:** Neon Serverless PostgreSQL Cloud Database
* 📧 **Email Delivery:** Resend HTTPS Transactional Email API

---

## 🏗️ System Architecture

```
                       ┌─────────────────────────────────────────┐
                       │           React 18 + Vite SPA           │
                       │           (Hosted on Vercel)            │
                       │  • Interactive Dashboard & Analytics    │
                       │  • Pure SVG Arc Churn Risk Gauge        │
                       │  • CSV/Excel Drag & Drop Upload Engine  │
                       └────────────────────┬────────────────────┘
                                            │
                                            │ HTTPS (JSON / Form-Data)
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │          FastAPI REST Backend           │
                       │          (Hosted on Railway)            │
                       │  • JWT Auth & Role-Based Access         │
                       │  • ML Inference Engine                  │
                       │  • Automated Playbook Recommendation    │
                       │  • Bulk File Processor (Pandas/OpenPyXL)│
                       └───────┬────────────┬────────────┬───────┘
                               │            │            │
            ┌──────────────────┘            │            └──────────────────┐
            ▼                               ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│  Neon PostgreSQL DB   │       │  Scikit-Learn Model   │       │   Resend Email API    │
│  • Users & RBAC Roles │       │  • Gradient Boosting  │       │  • 6-Digit OTP Mails  │
│  • Customer Profiles  │       │  • 7 Behavioral Input │       │  • Branded HTML UI    │
│  • Prediction History │       │    Features Pipeline  │       │  • Instant 0.5s Send  │
└───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

---

## 💡 Why I Built This & The Story Behind It

In subscription businesses (SaaS, Telecom, E-commerce), losing an existing customer costs 5x more than acquiring a new one. I wanted to build an end-to-end system where a Customer Success Manager or Business Owner can:
1. View all their customer accounts in one clean, responsive dashboard.
2. Get real-time ML risk scores (Critical, Moderate, Low).
3. Upload whole CSV or Excel spreadsheets (even with 1000+ rows) to batch predict churn instantly.
4. Know **exactly what action to take** to retain the customer before they cancel.

### 🤖 How the Frontend was Built (AI Prompt Engineering)
My core strengths are in **Machine Learning, Python, and Backend Development**. I had minimal prior frontend experience, so I built the entire React + Vite frontend using **iterative AI prompt engineering**:
* I designed the full system architecture, state management, API contracts, and component layouts.
* Used targeted prompts to generate the UI components, interactive animated donut charts, custom SVG churn gauges, and client-side caching.
* Solved real production challenges like upload timeouts, sliding JWT sessions, and instant data rendering purely through architectural prompt guidance.

---

## ✨ Key Features

* **⚡ Real-Time ML Churn Scoring:** Enter customer behavioral metrics to get an instant churn probability score (0% to 100%) powered by a trained Gradient Boosting ML model.
* **📋 Dynamic Retention Playbooks:** Generates actionable business strategies based on specific customer conditions (e.g., offer 15% annual renewal discount, assign priority technical support, schedule executive check-in).
* **📁 Bulk CSV & Excel Uploader:** Upload `.csv` or `.xlsx` files. The system checks column names on the client-side first, skips duplicates, and runs AI predictions for the entire file in parallel.
* **🌱 1-Click Demo Seed Generator:** Don't have sample data? Click the seed button on the dashboard to immediately populate 50 realistic accounts with live predictions.
* **📊 Visual Risk Analytics:** Real-time breakdown of customer churn risk across contract lengths (Month-to-month, 1 year, 2 years), payment channels, and regions (North America, Europe, Asia Pacific, LATAM, Africa).
* **🔐 Enterprise Authentication & Roles:** 
  * Passwords securely hashed with `bcrypt`.
  * Secure JWT session tokens with auto-refresh.
  * 6-digit passwordless Email OTP sign-in with branded HTML emails delivered via **Resend API**.
  * Role permissions: **Admin** (Full Access), **CSM** (Manage Customers & Run Predictions), and **Viewer** (Read-Only).

---

## 🛠️ Tech Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Frontend** | React 18, Vite, Radix UI Primitives, Axios, Custom CSS (Built via Prompt Engineering) |
| **Backend** | Python 3.11/3.13, FastAPI, SQLAlchemy 2.0 ORM, Pydantic v2, Uvicorn |
| **Machine Learning** | Scikit-Learn 1.7.1, GradientBoostingClassifier, Joblib, Pandas, NumPy |
| **Database** | Neon Serverless PostgreSQL (Production) / SQLite (Local Dev) |
| **Email Service** | Resend HTTPS Email API (Transactional 6-Digit OTP Delivery) |
| **Hosting & Cloud** | Railway (FastAPI Container) + Vercel (Edge CDN React Frontend) |

---

## 🔬 How the Machine Learning Model Works

The ML pipeline is trained on historical customer behavioral data using 7 core features:

```
Customer Input (7 Features)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ 1. Preprocessing (ColumnTransformer)                   │
│    • OneHotEncoding for: contract_type, payment_method,│
│      region                                            │
│    • Passthrough for: tenure, calls, late payments,    │
│      satisfaction score                                │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ 2. Model: GradientBoostingClassifier                   │
│    • 200 estimators, max depth 5, learning rate 0.05  │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
             Churn Probability (0% - 100%)
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
     Critical        Moderate           Low
     (> 75%)        (40% - 75%)       (< 40%)
```

### The 7 Model Input Features:

| Feature | Type | Range | What It Means |
| :--- | :--- | :--- | :--- |
| `tenure_months` | Integer | 1 to 72 | How many months the customer has been subscribed |
| `support_calls` | Integer | 0 to 20 | Number of times they called support (high calls = frustration) |
| `late_payments` | Integer | 0 to 12 | Number of late invoice payments |
| `satisfaction_score` | Float | 1.0 to 10.0 | Customer rating (scores ≤ 3.0 mean high churn risk) |
| `contract_type` | Categorical | Month-to-Month, 1-Year, 2-Year | Subscription term |
| `payment_method` | Categorical | Card, Digital Wallet, Bank Transfer | Payment channel |
| `region` | Categorical | North America, Europe, Asia, LATAM, Africa, South America | Geographic region |

---

## 🚀 How to Run Locally

Want to run this project on your own machine? Follow these simple steps:

### 1. Clone the repository
```bash
git clone https://github.com/krohitrkumar/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Set up the Backend
```bash
# Create and activate a Python virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI backend
python main.py
```
Your backend will be live at `http://localhost:8080`, and Swagger documentation at `http://localhost:8080/docs`.

### 3. Set up the Frontend
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open your browser and visit `http://localhost:5173` (or `http://localhost:3000`).

---

## 📁 Project Folder Structure

```
customer_churn_prediction/
│
├── backend/                       # FastAPI Backend
│   ├── main.py                    # API routing & CORS configuration
│   ├── database/                  # DB connection, security & config
│   ├── models/                    # SQLAlchemy models (User, Customer, Prediction)
│   ├── routes/                    # API endpoints (auth, customers, predict, analytics)
│   ├── schemas/                   # Pydantic request/response schemas
│   └── services/                  # ML prediction service & Resend/SMTP email service
│
├── frontend/                      # React 18 + Vite Frontend
│   ├── src/
│   │   ├── api/                   # API clients and endpoints
│   │   ├── components/            # Reusable UI components & charts
│   │   ├── context/               # AuthContext (JWT & login state)
│   │   ├── hooks/                 # Custom hooks (caching & notifications)
│   │   ├── pages/                 # Dashboard, Customers, Predict, Login, Register
│   │   └── styles/                # Global theme CSS and animations
│   └── package.json
│
├── ml_pipeline/                   # Machine Learning Pipeline
│   ├── train.py                   # Model training script
│   ├── data/                      # Dataset files
│   └── artifacts/                 # Saved model (churn_model.pkl)
│
├── main.py                        # Root server entrypoint
├── Procfile                       # Cloud deployment configuration
└── requirements.txt               # Python package dependencies
```

---

## 👨‍💻 Author

**Rohit Kumar**
* GitHub: [@krohitrkumar](https://github.com/krohitrkumar)
* Live Project: [Retentrix Churn Intelligence](https://customerchurnprediction-roan.vercel.app)

If you find this project helpful or interesting, feel free to give it a ⭐ on GitHub!
