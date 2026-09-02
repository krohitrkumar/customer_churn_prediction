# 🧠 Customer Churn Prediction & Retention System

Hey there! 👋 Welcome to my **Customer Churn Prediction System**. 

This is a complete full-stack web application that uses Machine Learning to predict whether a customer is likely to leave (churn) or stay. It doesn’t just give a probability score — it also suggests **automated retention playbooks** (action plans) based on why the customer might be unhappy (like too many support calls, low satisfaction, or payment delays).

---

## 🔗 Live Working Project

You can try the live application directly in your browser:

* 🌐 **Frontend Web App (Live on Vercel):** [https://customerchurnprediction-roan.vercel.app](https://customerchurnprediction-roan.vercel.app)
* ⚡ **Backend REST API (Live on Render):** [https://customer-churn-prediction-backend-y0q2.onrender.com](https://customer-churn-prediction-backend-y0q2.onrender.com)
* 📖 **Interactive Swagger API Docs:** [https://customer-churn-prediction-backend-y0q2.onrender.com/docs](https://customer-churn-prediction-backend-y0q2.onrender.com/docs)
* 🐘 **Database:** Neon PostgreSQL Cloud Database

---

## 💡 Why I Built This & The Story Behind It

In subscription businesses (SaaS, Telecom, E-commerce), losing an existing customer costs 5x more than acquiring a new one. I wanted to build an end-to-end system where a Customer Success Manager or Business Owner can:
1. View all their customer accounts in one clean dashboard.
2. Get real-time ML risk scores (Critical, Moderate, Low).
3. Upload whole CSV or Excel spreadsheets (even with 1000+ rows) to batch predict churn instantly.
4. Know **exactly what action to take** to retain the customer before they cancel.

### 🤖 How the Frontend was Built (AI Prompt Engineering)
My core strengths are in **Machine Learning, Python, and Backend Development**. I had minimal prior frontend experience, so I built the entire React + Vite frontend using **iterative AI prompt engineering**:
* I designed the system architecture, state flows, API requirements, and component layouts.
* Used targeted prompts to generate the UI components, interactive animated donut charts, custom SVG churn gauges, and client-side caching.
* Solved real production challenges like upload timeouts and instant data rendering purely through architectural prompt guidance.

---

## ✨ Key Features

* **⚡ Real-Time Churn Scoring:** Enter customer data to get an instant churn probability score (0% to 100%) powered by a trained Gradient Boosting ML model.
* **📋 Retention Playbooks:** Gives smart business recommendations based on profile conditions (e.g., offer 15% annual discount, assign priority technical support, schedule onboarding check-in).
* **📁 Bulk CSV & Excel Uploader:** Upload `.csv` or `.xlsx` files. The system checks column names on the client-side first, skips duplicates, and runs AI predictions for the entire file.
* **🌱 1-Click Demo Seed Generator:** Don't have sample data? Click the seed button on the dashboard to immediately populate 50 realistic accounts with live predictions.
* **📊 Visual Analytics:** Breakdown of customer churn risk across contract lengths (Month-to-month, 1 year, 2 years), payment channels, and regions (including LATAM, Asia Pacific, North America, Europe).
* **🔐 Secure Authentication & Roles:** 
  * Passwords encrypted with `bcrypt`.
  * Secure JWT tokens with 24-hour validity.
  * 6-digit passwordless Email OTP sign-in.
  * Role permissions (`admin`, `csm`, `user`).

---

## 🛠️ Tech Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Frontend** | React 18, Vite, Radix UI Primitives, Axios, Custom CSS (Built via Prompt Engineering) |
| **Backend** | Python 3.11, FastAPI, SQLAlchemy 2.0 ORM, Pydantic v2 |
| **Machine Learning** | Scikit-Learn 1.7.1, GradientBoostingClassifier, Joblib, Pandas, NumPy |
| **Database** | Neon Serverless PostgreSQL (Production) / SQLite (Local Dev) |
| **Hosting & Deploy** | Vercel (Frontend SPA) + Render (FastAPI Web Service) |

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
cd backend
uvicorn main:app --reload --port 8000
```
Your backend will be live at `http://localhost:8000`, and Swagger documentation at `http://localhost:8000/docs`.

### 3. Set up the Frontend
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open your browser and visit `http://localhost:3000` (or `http://localhost:5173`).

---

## 📁 Project Folder Structure

```
customer_churn_prediction/
│
├── backend/                       # FastAPI Backend
│   ├── main.py                    # Entrypoint & API routing
│   ├── database/                  # DB connection, security & config
│   ├── models/                    # SQLAlchemy models (User, Customer, Prediction)
│   ├── routes/                    # API endpoints (auth, customers, predict, analytics)
│   ├── schemas/                   # Pydantic request/response schemas
│   └── services/                  # ML prediction service & email OTP service
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
└── requirements.txt               # Python package dependencies
```

---

## 👨‍💻 Author

**Rohit Kumar**
* GitHub: [@krohitrkumar](https://github.com/krohitrkumar)
* Live Project: [Retentrix Churn Intelligence](https://customerchurnprediction-roan.vercel.app)

If you find this project helpful or interesting, feel free to give it a ⭐ on GitHub!
