# 🤖 AI Career Assistant

An AI-powered career assistant for Indian tech job seekers. Upload your resume, get instant skill gap analysis, a personalised 3-month roadmap, live job listings, and an AI career chatbot — all in one place.

---

## 📁 Project Structure

```
career_assistant/
│
├── main.py              # FastAPI backend — all API endpoints, auth, DB
├── agents.py            # LangChain AI agents (Skill Gap + Roadmap)
├── gemini_service.py    # Google Gemini 2.5 Flash wrapper
├── adzuna_service.py    # Adzuna Jobs API integration
├── resume_parser.py     # PDF resume text extractor
├── index.html           # Full frontend (single-page React app)
│
├── .env                 # Your secret keys (never commit this)
├── .env.example         # Template showing required env vars
├── .gitignore           # Ignores .env, db, cache files
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Resume Upload | Upload PDF → auto-extracts name, skills, education, experience |
| 🔍 Skill Gap Analysis | Compares your skills vs real job listings for any role |
| 🗺️ Learning Roadmap | AI-generated 3-month study plan with projects & resources |
| 💼 Live Jobs | Real job listings from Adzuna India API |
| 📊 Market Insights | Salary ranges, demand stats, top companies |
| 💬 Career Chatbot | Topic-restricted AI mentor for career & tech questions |
| 🔐 Full Auth | Register, Login, Email Verification, Change Password, Forgot Password |

---

## ⚙️ Setup & Installation

### Step 1 — Clone the project
```bash
git clone https://github.com/yourusername/career-assistant.git
cd career-assistant
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and fill in your API keys (see API Keys section below).

### Step 5 — Run the server
```bash
uvicorn main:app --reload
```

### Step 6 — Open the app
- **Frontend:** Open `index.html` in your browser
- **API Docs:** Visit `http://localhost:8000/docs`

---

## 🔑 API Keys Required

### 1. Google Gemini API Key (Required)
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → Create API Key
3. Copy and paste into `.env` as `GEMINI_API_KEY`

### 2. Adzuna Jobs API (Required for job listings)
1. Sign up free at [developer.adzuna.com](https://developer.adzuna.com)
2. Create an app → copy **App ID** and **App Key**
3. Add to `.env` as `ADZUNA_APP_ID` and `ADZUNA_APP_KEY`

### 3. Gmail App Password (Required for emails)
1. Enable **2-Step Verification** on your Google account
2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
3. Create app password → copy the 16-digit code
4. Add to `.env` as `SMTP_USER` (your Gmail) and `SMTP_PASS` (16-digit code)

---

## 🔐 Auth Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Create account (sends verification email) |
| POST | `/auth/login/json` | Login → returns JWT token |
| GET | `/auth/me` | Get current user info |
| POST | `/auth/logout` | Logout (revokes token) |
| GET | `/auth/verify-email?token=...` | Verify email from link |
| POST | `/auth/resend-verification` | Resend verification email |
| POST | `/auth/change-password` | Change password (requires old password) |
| POST | `/auth/forgot-password` | Send reset link to email |
| POST | `/auth/reset-password` | Reset password using token |
| POST | `/auth/refresh` | Get fresh JWT token |

### Password Requirements
- Minimum 8 characters
- At least 1 uppercase letter (A-Z)
- At least 1 lowercase letter (a-z)
- At least 1 number (0-9)
- At least 1 special character (!@#$%^&*)

---

## 🧪 Testing the API

Open `http://localhost:8000/docs` for interactive Swagger UI.

**Quick test with curl:**
```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@gmail.com","password":"Test@1234"}'

# Login
curl -X POST http://localhost:8000/auth/login/json \
  -H "Content-Type: application/json" \
  -d '{"email":"test@gmail.com","password":"Test@1234"}'
```

---

## 🚀 Deployment (Free)

### Deploy to Render
1. Push your code to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set **Build Command:** `pip install -r requirements.txt`
5. Set **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 10000`
6. Add all environment variables from `.env` in the Render dashboard
7. Deploy → copy your live URL

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Database | SQLite + SQLAlchemy |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| AI | Google Gemini 2.5 Flash |
| Agents | LangChain ReAct Agents |
| Jobs API | Adzuna India |
| PDF Parsing | pdfplumber |
| Email | smtplib (Gmail SMTP) |
| Frontend | React 18 (via CDN, single HTML file) |

---

## 📞 Support

If you face any issues:
1. Check `http://localhost:8000/docs` for API errors
2. Check terminal logs for `[Email]` or `[Agents]` messages
3. Verify all `.env` keys are filled correctly
