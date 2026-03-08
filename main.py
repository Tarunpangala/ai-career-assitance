import warnings
warnings.filterwarnings("ignore", message="Field name .* shadows an attribute in parent", category=UserWarning)

from fastapi import FastAPI, UploadFile, File, Query, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, validator
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import shutil, asyncio, re, json, os, secrets, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

from gemini_service import ask_gemini, ask_gemini_async
from resume_parser  import extract_text
from adzuna_service import get_jobs
from agents import (
    compute_confidence,
    score_resume,
    run_skill_gap_agent_async,
    run_roadmap_agent_async,
)

# ── Database ──────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./career_assistant.db"
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base         = declarative_base()


# ── Models ────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    id                 = Column(Integer, primary_key=True, index=True)
    name               = Column(String(100), nullable=False)
    email              = Column(String(150), unique=True, index=True, nullable=False)
    hashed_pw          = Column(String(255), nullable=False)
    is_admin           = Column(Integer, default=0)
    is_verified        = Column(Integer, default=0)        # 1 = verified
    otp_code           = Column(String(6),   nullable=True) # 6-digit OTP
    otp_expiry         = Column(DateTime,    nullable=True) # OTP expires in 10 min
    otp_attempts       = Column(Integer,     default=0)     # wrong attempt counter
    reset_token        = Column(String(100), nullable=True) # forgot password token
    reset_token_expiry = Column(DateTime,    nullable=True) # reset token expiry
    created_at         = Column(DateTime, default=datetime.utcnow)


class UserData(Base):
    __tablename__ = "user_data"
    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, unique=True, index=True, nullable=False)
    resume_text   = Column(Text, default="")
    profile_json  = Column(Text, default="{}")
    analyses_json = Column(Text, default="{}")
    roadmaps_json = Column(Text, default="{}")
    chat_json     = Column(Text, default="[]")
    updated_at    = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_ud(db, user_id):
    ud = db.query(UserData).filter(UserData.user_id == user_id).first()
    if not ud:
        ud = UserData(user_id=user_id)
        db.add(ud); db.commit(); db.refresh(ud)
    return ud


def save_ud(db, ud):
    ud.updated_at = datetime.utcnow()
    db.commit()


# ── Auth ──────────────────────────────────────────────────────────────
SECRET_KEY   = os.getenv("JWT_SECRET", "careerbot-secret-2025-change-me")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
ALGORITHM    = "HS256"
TOKEN_MINS   = 60 * 24 * 7

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2  = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def hash_pw(pw):         return pwd_ctx.hash(pw[:72])
def verify_pw(p, h):     return pwd_ctx.verify(p[:72], h)
def make_token(uid):     return jwt.encode({"sub": str(uid), "exp": datetime.utcnow() + timedelta(minutes=TOKEN_MINS)}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)):
    if not token: return None
    if token in REVOKED_TOKENS: return None   # ✅ revoked token check
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if uid is None: return None
    except JWTError:
        return None
    return db.query(User).filter(User.id == int(uid)).first()


def require_user(u=Depends(get_current_user)):
    if not u: raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return u


# ── In-memory caches ──────────────────────────────────────────────────
RESUME_STORE    = {}
PROFILE_STORE   = {}
ANALYZE_CACHE   = {}
ROADMAP_CACHE   = {}
CHAT_CALL_COUNT = 0
REVOKED_TOKENS  = set()   # ✅ logout / token revocation

def _uid(user): return str(user.id) if user else "anon"

# ── Email config ──────────────────────────────────────────────────────
SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")


def send_email(to: str, subject: str, html_body: str) -> bool:
    """Send email via SMTP. Returns True on success."""
    if not SMTP_USER or not SMTP_PASS:
        print(f"[Email] SMTP not configured. Would send to {to}: {subject}")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = to
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to, msg.as_string())
        return True
    except Exception as e:
        print(f"[Email] Failed to send to {to}: {e}")
        return False


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="AI Career Assistant API", version="10.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup_checks():
    missing = [k for k in ["GEMINI_API_KEY", "JWT_SECRET", "ADMIN_SECRET"] if not os.getenv(k)]
    if missing:
        print(f"[Startup] ⚠️  Missing env vars: {', '.join(missing)}")
    else:
        print("[Startup] ✅ All environment variables set")


# ── Static chat answers ───────────────────────────────────────────────
STATIC = {
    "dsa": "🧠 **DSA Interview Prep**\n• Arrays → Strings → LinkedList → Trees → Graphs → DP\n• Practice 2–3 LeetCode problems daily (Easy → Medium → Hard)\n• neetcode.io has the best free structured roadmap\n📌 Tip: Solve 100 LeetCode Mediums before applying to product companies.",
    "system design": "🏗️ **System Design**\n• Core: Load Balancing, Caching, SQL vs NoSQL, APIs, CDN\n• Study: URL Shortener, Twitter Feed, WhatsApp\n📌 Tip: Watch Gaurav Sen on YouTube — best free system design content.",
    "git": "🔧 **Git Essentials**\n• Commands: clone, add, commit, push, pull, branch, merge, rebase\n• Use feature branches → PR → review → merge\n📌 Tip: Write good commit messages: 'feat: add login page'",
    "python basics": "🐍 **Learn Python Fast**\n• Week 1: Variables, loops, functions, lists, dicts\n• Week 2: OOP, file handling, error handling\n• Week 3+: Projects — calculator, todo app, web scraper\n📌 Tip: CS50P (Harvard, free on edX) is the best structured course.",
    "resume tips": "📄 **Resume Tips**\n• 1 page only — ATS-friendly template (no tables/columns/images)\n• Format: 'Built X using Y which resulted in Z'\n📌 Tip: Score free at resumeworded.com before applying.",
    "interview tips": "💼 **Interview Prep**\n• Tech: DSA + System Design + project deep-dive\n• HR: STAR method — Situation, Task, Action, Result\n📌 Tip: 5 mock interviews on Pramp (free) cuts nervousness fast.",
    "free courses": "🎓 **Free Resources**\n• CS Fundamentals: CS50 — edx.org/cs50\n• Web Dev: The Odin Project — theodinproject.com\n• DSA: NeetCode — neetcode.io\n📌 Tip: Finish ONE course fully. A GitHub project beats a certificate.",
    "linkedin": "🔗 **LinkedIn Tips**\n• Headline: 'B.Tech CS | Python | React | Seeking SDE Roles 2025'\n• Add all projects with GitHub links\n📌 Tip: Turn on 'Open to Work' — recruiters filter for this.",
    "open source": "🌐 **Open Source**\n• Start with 'good first issue' label on GitHub\n• Good projects: freeCodeCamp, first-contributions, EddieHub\n📌 Tip: One merged PR to a known repo beats 10 toy projects.",
}

def get_static(msg):
    ml = msg.lower()
    if "system design" in ml: return STATIC["system design"]
    if any(k in ml for k in ["dsa","data structure","algorithm","leetcode"]): return STATIC["dsa"]
    if "git" in ml and "github" not in ml: return STATIC["git"]
    if "python" in ml and any(k in ml for k in ["learn","start","beginner","basic"]): return STATIC["python basics"]
    if "resume tip" in ml or ("resume" in ml and "tip" in ml): return STATIC["resume tips"]
    if "interview tip" in ml or ("interview" in ml and "prepare" in ml): return STATIC["interview tips"]
    if "free course" in ml or "best resource" in ml: return STATIC["free courses"]
    if "linkedin" in ml: return STATIC["linkedin"]
    if "open source" in ml or "contribute" in ml: return STATIC["open source"]
    return None


# ── Helpers ───────────────────────────────────────────────────────────
def extract_score(text):
    for p in [r'MATCH SCORE:\s*(\d+)\s*/\s*100', r'(\d+)\s*/\s*100']:
        m = re.search(p, text, re.IGNORECASE)
        if m: return int(m.group(1))
    return 75


def clean_json(raw):
    raw = raw.strip()
    if '```' in raw:
        parts = raw.split('```')
        raw = parts[1] if len(parts) >= 2 else raw
        raw = raw.lstrip('json').strip()
    return raw


def extract_json_block(text: str, key: str, opener: str) -> str:
    """
    Extract a JSON object {} or array [] from text after a given key label.
    Uses bracket counting so nested arrays/objects don't break parsing.
    Returns the raw JSON string or "" if not found.
    """
    marker = text.find(key + ":")
    if marker == -1:
        return ""
    start = text.find(opener, marker)
    if start == -1:
        return ""
    closer = "}" if opener == "{" else "]"
    depth  = 0
    for i, ch in enumerate(text[start:], start):
        if   ch == opener:  depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return ""


# Role-specific project fallbacks — shown when Gemini JSON parsing fails
ROLE_PROJECTS = {
    "full stack developer": [
        {"title":"Full Stack E-Commerce App","description":"Build a complete online store with React frontend, Node.js backend, and PostgreSQL database.","tech":["React","Node.js","PostgreSQL","Tailwind"],"icon":"🛒","difficulty":"Beginner","duration":"3-4 weeks","steps":["Step 1: Design DB schema, set up Node/Express + PostgreSQL","Step 2: Build REST API (products, cart, orders)","Step 3: Build React frontend with routing and state","Step 4: Add auth (JWT) and deploy to Render + Vercel"]},
        {"title":"Real-Time Chat Application","description":"WebSocket-based chat app with rooms, online presence, and message history.","tech":["React","Node.js","Socket.io","MongoDB"],"icon":"💬","difficulty":"Intermediate","duration":"3-4 weeks","steps":["Step 1: Set up Socket.io server with room management","Step 2: Build React chat UI with live updates","Step 3: Add MongoDB for message persistence","Step 4: Deploy backend to Railway, frontend to Vercel"]},
        {"title":"Developer Portfolio + Blog CMS","description":"Personal portfolio with a custom headless CMS to write and publish blog posts.","tech":["Next.js","TypeScript","PostgreSQL","Tailwind"],"icon":"🚀","difficulty":"Advanced","duration":"4-5 weeks","steps":["Step 1: Set up Next.js with TypeScript and Tailwind","Step 2: Build CMS admin panel with CRUD for posts","Step 3: Add markdown rendering, SEO, and image uploads","Step 4: Deploy to Vercel, write a blog post about it"]},
    ],
    "data scientist": [
        {"title":"End-to-End ML Pipeline","description":"Predict house prices or loan defaults with a full pipeline from raw data to deployed API.","tech":["Python","Scikit-learn","FastAPI","Pandas"],"icon":"🏠","difficulty":"Beginner","duration":"2-3 weeks","steps":["Step 1: EDA and feature engineering on Kaggle dataset","Step 2: Train and compare 3 models, pick the best","Step 3: Wrap model in FastAPI endpoint","Step 4: Deploy to Render, add a simple UI"]},
        {"title":"NLP Sentiment Analyser","description":"Analyse product reviews or tweets for sentiment using transformers.","tech":["Python","HuggingFace","Streamlit","Pandas"],"icon":"💬","difficulty":"Intermediate","duration":"3-4 weeks","steps":["Step 1: Collect and clean dataset (scrape or use existing)","Step 2: Fine-tune BERT or use pre-trained model","Step 3: Build Streamlit dashboard with charts","Step 4: Deploy to Streamlit Cloud"]},
        {"title":"ML Model Monitoring Dashboard","description":"Track model drift, data quality, and prediction confidence in production.","tech":["Python","MLflow","Grafana","Docker"],"icon":"📊","difficulty":"Advanced","duration":"4-5 weeks","steps":["Step 1: Train baseline model, log with MLflow","Step 2: Simulate data drift scenarios","Step 3: Build monitoring metrics and alerts","Step 4: Containerise with Docker, write a blog post"]},
    ],
    "backend developer": [
        {"title":"REST API with Auth & Rate Limiting","description":"Production-ready API with JWT auth, role-based access, and Redis rate limiting.","tech":["Python","FastAPI","PostgreSQL","Redis"],"icon":"🔐","difficulty":"Beginner","duration":"2-3 weeks","steps":["Step 1: Set up FastAPI + PostgreSQL with SQLAlchemy","Step 2: Add JWT auth with refresh tokens","Step 3: Add Redis rate limiting and caching","Step 4: Write API docs and deploy to Render"]},
        {"title":"Microservices with Message Queue","description":"Two services communicating via RabbitMQ or Kafka for async order processing.","tech":["Python","FastAPI","Kafka","Docker"],"icon":"⚙️","difficulty":"Intermediate","duration":"4-5 weeks","steps":["Step 1: Design service boundaries and API contracts","Step 2: Build order service and notification service","Step 3: Add Kafka for async communication","Step 4: Orchestrate with Docker Compose"]},
        {"title":"GitHub CI/CD Pipeline","description":"Fully automated test → build → deploy pipeline for a backend API.","tech":["GitHub Actions","Docker","PostgreSQL","Python"],"icon":"🚀","difficulty":"Advanced","duration":"3-4 weeks","steps":["Step 1: Write unit + integration tests","Step 2: Set up GitHub Actions workflow","Step 3: Add Docker build and push to registry","Step 4: Auto-deploy to cloud on merge to main"]},
    ],
    "frontend developer": [
        {"title":"Component Library with Storybook","description":"Build a reusable React component library with docs, tests, and npm publish.","tech":["React","TypeScript","Storybook","Tailwind"],"icon":"🎨","difficulty":"Beginner","duration":"3-4 weeks","steps":["Step 1: Set up React + TypeScript + Tailwind","Step 2: Build 10+ reusable components","Step 3: Add Storybook docs and Jest tests","Step 4: Publish to npm, add to your portfolio"]},
        {"title":"Real-Time Dashboard with WebSockets","description":"Live analytics dashboard with charts that update in real-time.","tech":["React","TypeScript","Socket.io","Recharts"],"icon":"📊","difficulty":"Intermediate","duration":"3-4 weeks","steps":["Step 1: Set up React with TypeScript","Step 2: Connect to WebSocket for live data","Step 3: Build Recharts visualisations","Step 4: Add dark mode, deploy to Vercel"]},
        {"title":"Progressive Web App (PWA)","description":"Offline-capable PWA with push notifications and app-like feel.","tech":["React","TypeScript","Service Workers","IndexedDB"],"icon":"📱","difficulty":"Advanced","duration":"4-5 weeks","steps":["Step 1: Set up React PWA with manifest and service worker","Step 2: Add offline caching strategy","Step 3: Implement push notifications","Step 4: Score 90+ on Lighthouse, deploy to Vercel"]},
    ],
    "devops engineer": [
        {"title":"Kubernetes Cluster on AWS","description":"Deploy a 3-tier app to EKS with auto-scaling, monitoring and alerts.","tech":["Kubernetes","AWS","Terraform","Helm"],"icon":"☸️","difficulty":"Beginner","duration":"3-4 weeks","steps":["Step 1: Provision EKS cluster with Terraform","Step 2: Deploy app with Helm charts","Step 3: Add HPA for auto-scaling","Step 4: Set up Prometheus + Grafana monitoring"]},
        {"title":"Full CI/CD Pipeline with GitOps","description":"ArgoCD-based GitOps pipeline — any push to main auto-deploys to production.","tech":["GitHub Actions","ArgoCD","Docker","Kubernetes"],"icon":"🔄","difficulty":"Intermediate","duration":"3-4 weeks","steps":["Step 1: Containerise app and push to Docker Hub","Step 2: Set up ArgoCD on cluster","Step 3: Configure GitHub Actions to update image tags","Step 4: Test rollback and blue-green deploy"]},
        {"title":"Infrastructure as Code Portfolio","description":"Full AWS infrastructure for a production app — VPC, ECS, RDS, CloudFront.","tech":["Terraform","AWS","Docker","Python"],"icon":"🏗️","difficulty":"Advanced","duration":"5-6 weeks","steps":["Step 1: Design VPC with public/private subnets","Step 2: ECS Fargate for containers, RDS for database","Step 3: CloudFront + WAF for the frontend","Step 4: Add cost alerts, document everything in README"]},
    ],
    "ml engineer": [
        {"title":"Model Serving API with FastAPI","description":"Deploy a trained ML model as a scalable REST API with monitoring.","tech":["Python","FastAPI","Docker","MLflow"],"icon":"🤖","difficulty":"Beginner","duration":"2-3 weeks","steps":["Step 1: Train model, serialise with joblib/ONNX","Step 2: Build FastAPI prediction endpoint","Step 3: Add input validation and logging","Step 4: Containerise and deploy to Render"]},
        {"title":"Feature Store + Training Pipeline","description":"Automated ML pipeline: data → features → training → registry → serving.","tech":["Python","Airflow","MLflow","PostgreSQL"],"icon":"⚙️","difficulty":"Intermediate","duration":"4-5 weeks","steps":["Step 1: Build feature engineering pipeline with Airflow","Step 2: Store features in PostgreSQL feature store","Step 3: Auto-trigger training when new data arrives","Step 4: Register best model in MLflow, serve via API"]},
        {"title":"LLM-Powered App with RAG","description":"Chat with your documents using LangChain, embeddings and a vector database.","tech":["Python","LangChain","ChromaDB","FastAPI"],"icon":"🧠","difficulty":"Advanced","duration":"4-6 weeks","steps":["Step 1: Set up document ingestion and chunking","Step 2: Generate embeddings and store in ChromaDB","Step 3: Build RAG chain with LangChain","Step 4: Add streaming API, deploy and demo publicly"]},
    ],
}

def parse_profile(text):
    email_m = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)
    phone_m = re.search(r'(\+?\d[\d\s\-\.]{8,14}\d)', text)
    SKIP    = ["resume","curriculum","vitae","profile","contact","email","phone","mobile",
               "address","linkedin","github","portfolio","objective","summary","about",
               "skills","education","experience","projects"]
    name = "Not specified"
    for line in [l.strip() for l in text.split('\n') if l.strip()][:10]:
        words = line.split()
        if (2 <= len(words) <= 5 and not any(c.isdigit() for c in line)
                and not any(w in line.lower() for w in SKIP)
                and not re.search(r'[|•:@/\\()]', line)):
            name = line.title(); break

    prompt = (
        "You are a resume parser. Return ONLY valid JSON.\n"
        '{"education":"Degree and branch","college":"Full college name","cgpa":"CGPA/percentage",'
        '"experience":"Fresher or X years","skills":["tech skills only, min 5 max 15"]}\n'
        f"Resume:\n{text[:2500]}"
    )
    try:
        data   = json.loads(clean_json(ask_gemini(prompt)))
        skills = [s for s in (data.get("skills", []) if isinstance(data.get("skills", []), list) else [])
                  if s and s.lower() not in {"communication","teamwork","leadership",
                                              "problem solving","critical thinking","time management"}][:15]
        return {"name": name, "email": email_m.group(0) if email_m else "Not specified",
                "phone": phone_m.group(0).strip() if phone_m else "Not specified",
                "education": data.get("education","Not specified"),
                "college":   data.get("college",  "Not specified"),
                "cgpa":      data.get("cgpa",      "Not specified"),
                "experience":data.get("experience","Fresher"), "skills": skills}
    except Exception:
        KNOWN  = ["Python","Java","JavaScript","TypeScript","React","Angular","Vue","Node.js",
                  "Django","Flask","FastAPI","SQL","MySQL","PostgreSQL","MongoDB","Redis",
                  "AWS","Azure","GCP","Docker","Kubernetes","Linux","Git","Machine Learning",
                  "Deep Learning","TensorFlow","PyTorch","Pandas","NumPy","HTML","CSS","Tailwind"]
        skills = [s for s in KNOWN if re.search(re.escape(s), text, re.IGNORECASE)][:12]
        edu_m  = re.search(r'(B\.?Tech|B\.?E\.?|B\.?Sc|M\.?Tech|M\.?Sc|MBA|BCA|MCA|Ph\.?D)[^\n,]{0,60}', text, re.IGNORECASE)
        exp_m  = re.search(r'(\d+\.?\d*)\s*\+?\s*years?', text, re.IGNORECASE)
        return {"name": name, "email": email_m.group(0) if email_m else "Not specified",
                "phone": phone_m.group(0).strip() if phone_m else "Not specified",
                "education":  edu_m.group(0).strip() if edu_m else "Not specified",
                "college":    "Not specified", "cgpa": "Not specified",
                "experience": f"{exp_m.group(1)} years" if exp_m else "Fresher",
                "skills":     skills}


def get_role_projects(role: str, fallback_role: str = None) -> list:
    """Return role-specific project suggestions."""
    role_l = role.lower().strip()
    for key, projects in ROLE_PROJECTS.items():
        if key in role_l or role_l in key:
            return projects
    if fallback_role:
        for key, projects in ROLE_PROJECTS.items():
            if key in fallback_role.lower():
                return projects
    # Generic fallback
    return [
        {"title":f"{role} Portfolio Project","description":f"Build a production-ready {role} project to showcase your skills.","tech":["Python","React","PostgreSQL","Docker"],"icon":"💼","difficulty":"Beginner","duration":"3-4 weeks","steps":["Step 1: Plan features and set up the project structure","Step 2: Build the core functionality","Step 3: Add tests, documentation and a good README","Step 4: Deploy publicly and add to your portfolio"]},
        {"title":f"{role} API + Dashboard","description":f"REST API with a frontend dashboard relevant to {role} work.","tech":["FastAPI","React","PostgreSQL","Tailwind"],"icon":"📊","difficulty":"Intermediate","duration":"3-4 weeks","steps":["Step 1: Design API endpoints and database schema","Step 2: Build backend with FastAPI","Step 3: Build React dashboard consuming the API","Step 4: Add auth, deploy to Render + Vercel"]},
        {"title":f"{role} Capstone","description":f"End-to-end {role} showcase that solves a real problem.","tech":["Docker","GitHub Actions","Cloud","Git"],"icon":"🚀","difficulty":"Advanced","duration":"4-6 weeks","steps":["Step 1: Pick a real problem and design the solution","Step 2: Build MVP in 2 weeks","Step 3: Add CI/CD pipeline and monitoring","Step 4: Write a blog post, publish on LinkedIn"]},
    ]


    email_m = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)
    phone_m = re.search(r'(\+?\d[\d\s\-\.]{8,14}\d)', text)
    SKIP    = ["resume","curriculum","vitae","profile","contact","email","phone","mobile",
               "address","linkedin","github","portfolio","objective","summary","about",
               "skills","education","experience","projects"]
    name = "Not specified"
    for line in [l.strip() for l in text.split('\n') if l.strip()][:10]:
        words = line.split()
        if (2 <= len(words) <= 5 and not any(c.isdigit() for c in line)
                and not any(w in line.lower() for w in SKIP)
                and not re.search(r'[|•:@/\\()]', line)):
            name = line.title(); break

    prompt = (
        "You are a resume parser. Return ONLY valid JSON.\n"
        '{"education":"Degree and branch","college":"Full college name","cgpa":"CGPA/percentage",'
        '"experience":"Fresher or X years","skills":["tech skills only, min 5 max 15"]}\n'
        f"Resume:\n{text[:2500]}"
    )
    try:
        data   = json.loads(clean_json(ask_gemini(prompt)))
        skills = [s for s in (data.get("skills", []) if isinstance(data.get("skills", []), list) else [])
                  if s and s.lower() not in {"communication","teamwork","leadership",
                                              "problem solving","critical thinking","time management"}][:15]
        return {"name": name, "email": email_m.group(0) if email_m else "Not specified",
                "phone": phone_m.group(0).strip() if phone_m else "Not specified",
                "education": data.get("education","Not specified"),
                "college":   data.get("college",  "Not specified"),
                "cgpa":      data.get("cgpa",      "Not specified"),
                "experience":data.get("experience","Fresher"), "skills": skills}
    except Exception:
        KNOWN  = ["Python","Java","JavaScript","TypeScript","React","Angular","Vue","Node.js",
                  "Django","Flask","FastAPI","SQL","MySQL","PostgreSQL","MongoDB","Redis",
                  "AWS","Azure","GCP","Docker","Kubernetes","Linux","Git","Machine Learning",
                  "Deep Learning","TensorFlow","PyTorch","Pandas","NumPy","HTML","CSS","Tailwind"]
        skills = [s for s in KNOWN if re.search(re.escape(s), text, re.IGNORECASE)][:12]
        edu_m  = re.search(r'(B\.?Tech|B\.?E\.?|B\.?Sc|M\.?Tech|M\.?Sc|MBA|BCA|MCA|Ph\.?D)[^\n,]{0,60}', text, re.IGNORECASE)
        exp_m  = re.search(r'(\d+\.?\d*)\s*\+?\s*years?', text, re.IGNORECASE)
        return {"name": name, "email": email_m.group(0) if email_m else "Not specified",
                "phone": phone_m.group(0).strip() if phone_m else "Not specified",
                "education":  edu_m.group(0).strip() if edu_m else "Not specified",
                "college":    "Not specified", "cgpa": "Not specified",
                "experience": f"{exp_m.group(1)} years" if exp_m else "Fresher",
                "skills":     skills}


# ════════════════════════════════════════════════════════════════════
#  AUTH
# ════════════════════════════════════════════════════════════════════

# ── OTP store: { email -> {otp, expiry, attempts, pending_user_data} } ──
OTP_STORE = {}
OTP_EXPIRY_MINUTES = 10
OTP_MAX_ATTEMPTS   = 5


def generate_otp() -> str:
    """Generate a secure 6-digit OTP."""
    import random
    return str(random.SystemRandom().randint(100000, 999999))


def send_otp_email(to: str, name: str, otp: str) -> bool:
    return send_email(
        to       = to,
        subject  = "🔐 Your AI Career Assistant OTP",
        html_body= f"""
        <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:36px;
                    background:#f8faff;border-radius:12px;border:1px solid #e2edff">
          <div style="text-align:center;margin-bottom:24px">
            <div style="font-size:48px">🤖</div>
            <h2 style="color:#1d6ae5;margin:8px 0">AI Career Assistant</h2>
          </div>
          <p style="color:#333;font-size:15px">Hi <b>{name}</b>,</p>
          <p style="color:#555;font-size:14px;line-height:1.6">
            Use the OTP below to verify your email and activate your account.
          </p>
          <div style="text-align:center;margin:28px 0">
            <div style="display:inline-block;background:#1d6ae5;color:white;
                        font-size:36px;font-weight:900;letter-spacing:10px;
                        padding:18px 36px;border-radius:12px;
                        font-family:monospace">
              {otp}
            </div>
          </div>
          <p style="color:#e74c3c;font-size:13px;text-align:center;font-weight:600">
            ⏰ This OTP expires in {OTP_EXPIRY_MINUTES} minutes
          </p>
          <p style="color:#999;font-size:12px;text-align:center;margin-top:20px">
            If you didn't request this, ignore this email.
          </p>
        </div>"""
    )


class AuthReq(BaseModel):
    name:     str = ""
    email:    str
    password: str

    @validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character")
        return v

    @validator("email")
    def email_format(cls, v):
        v = v.strip().lower()
        if not re.match(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$", v):
            raise ValueError("Invalid email format")
        return v


# ── STEP 1: Register → send OTP (account NOT created yet) ─────────────

@app.post("/auth/register", tags=["Auth"])
def register(req: AuthReq, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(400, "Email already registered")

    otp    = generate_otp()
    expiry = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)

    # Store registration data temporarily — NOT saved to DB yet
    OTP_STORE[req.email] = {
        "otp":      otp,
        "expiry":   expiry,
        "attempts": 0,
        "name":     req.name.strip(),
        "password": req.password,   # plain password — hashed on account creation
    }

    sent = send_otp_email(req.email, req.name.strip(), otp)

    return {
        "message":    f"OTP sent to {req.email}. Enter it to complete registration.",
        "email":      req.email,
        "expires_in": f"{OTP_EXPIRY_MINUTES} minutes",
        "email_sent": sent,
        # Dev helper: show OTP in response if SMTP not configured
        **({"otp_preview": otp} if not sent else {}),
    }


# ── STEP 2: Verify OTP → create account ───────────────────────────────

class OTPVerifyReq(BaseModel):
    email: str
    otp:   str


@app.post("/auth/verify-otp", tags=["Auth"])
def verify_otp(req: OTPVerifyReq, db: Session = Depends(get_db)):
    email = req.email.strip().lower()
    entry = OTP_STORE.get(email)

    # OTP not found
    if not entry:
        raise HTTPException(400, "No OTP found for this email. Please register again.")

    # OTP expired
    if datetime.utcnow() > entry["expiry"]:
        OTP_STORE.pop(email, None)
        raise HTTPException(400, "OTP has expired. Please register again to get a new OTP.")

    # Too many wrong attempts
    if entry["attempts"] >= OTP_MAX_ATTEMPTS:
        OTP_STORE.pop(email, None)
        raise HTTPException(400, "Too many wrong attempts. Please register again.")

    # Wrong OTP
    if req.otp.strip() != entry["otp"]:
        entry["attempts"] += 1
        left = OTP_MAX_ATTEMPTS - entry["attempts"]
        raise HTTPException(400, f"Incorrect OTP. {left} attempt(s) remaining.")

    # ✅ OTP is correct — now create the account
    OTP_STORE.pop(email, None)

    # Double-check email not registered while OTP was pending
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already registered.")

    u = User(
        name        = entry["name"],
        email       = email,
        hashed_pw   = hash_pw(entry["password"]),
        is_verified = 1,   # verified immediately via OTP
    )
    db.add(u); db.commit(); db.refresh(u)
    get_or_create_ud(db, u.id)

    return {
        "message":      "✅ Account created successfully! You can now log in.",
        "access_token": make_token(u.id),
        "token_type":   "bearer",
        "is_verified":  True,
        "is_admin":     bool(u.is_admin),
        "user":         {"id": u.id, "name": u.name, "email": u.email, "is_admin": bool(u.is_admin)},
    }


# ── Resend OTP ─────────────────────────────────────────────────────────

class ResendOTPReq(BaseModel):
    email: str


@app.post("/auth/resend-otp", tags=["Auth"])
def resend_otp(req: ResendOTPReq):
    email = req.email.strip().lower()
    entry = OTP_STORE.get(email)

    if not entry:
        raise HTTPException(400, "No pending registration for this email. Please register first.")

    otp    = generate_otp()
    expiry = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)

    entry["otp"]      = otp
    entry["expiry"]   = expiry
    entry["attempts"] = 0

    sent = send_otp_email(email, entry["name"], otp)

    return {
        "message":  "New OTP sent. Please check your inbox.",
        "email_sent": sent,
        **({"otp_preview": otp} if not sent else {}),
    }


@app.post("/auth/login/json", tags=["Auth"])
def login_json(req: AuthReq, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == req.email).first()
    if not u or not verify_pw(req.password, u.hashed_pw):
        raise HTTPException(401, "Invalid email or password")
    return {
        "access_token": make_token(u.id),
        "token_type":   "bearer",
        "is_verified":  bool(u.is_verified),
        "is_admin":     bool(u.is_admin),
        "user":         {"id": u.id, "name": u.name, "email": u.email, "is_admin": bool(u.is_admin)}
    }


@app.get("/auth/me", tags=["Auth"])
def me(u=Depends(require_user)):
    return {"id": u.id, "name": u.name, "email": u.email,
            "is_verified": bool(u.is_verified), "is_admin": bool(u.is_admin)}

# ── ✅ LOGOUT / TOKEN REVOCATION ──────────────────────────────────────

@app.post("/auth/logout", tags=["Auth"])
def logout(token: str = Depends(oauth2), u=Depends(require_user)):
    if token:
        REVOKED_TOKENS.add(token)
    return {"message": "✅ Logged out successfully"}


# ── ✅ CHANGE PASSWORD ────────────────────────────────────────────────

class ChangePasswordReq(BaseModel):
    old_password: str
    new_password: str

    @validator("new_password")
    def new_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Must contain an uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Must contain a lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Must contain a number")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Must contain a special character")
        return v


@app.post("/auth/change-password", tags=["Auth"])
def change_password(req: ChangePasswordReq, u=Depends(require_user), db: Session = Depends(get_db)):
    if not verify_pw(req.old_password, u.hashed_pw):
        raise HTTPException(400, "Current password is incorrect")
    if req.old_password == req.new_password:
        raise HTTPException(400, "New password must be different from current password")
    u.hashed_pw = hash_pw(req.new_password)
    db.commit()
    send_email(
        to       = u.email,
        subject  = "🔒 Your password was changed",
        html_body= f"""
        <div style="font-family:Arial,sans-serif;max-width:500px;margin:auto;padding:30px">
          <h2 style="color:#4F46E5">Password Changed</h2>
          <p>Hi <b>{u.name}</b>, your password was just changed.</p>
          <p>If you did not do this, please contact support immediately.</p>
        </div>"""
    )
    return {"message": "✅ Password changed successfully"}


# ── ✅ FORGOT PASSWORD ────────────────────────────────────────────────

class ForgotPasswordReq(BaseModel):
    email: str


class ResetPasswordReq(BaseModel):
    token:        str
    new_password: str

    @validator("new_password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Must contain an uppercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Must contain a number")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Must contain a special character")
        return v


@app.post("/auth/forgot-password", tags=["Auth"])
def forgot_password(req: ForgotPasswordReq, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == req.email.strip().lower()).first()
    # Always return same message to prevent email enumeration
    if not u:
        return {"message": "If that email exists, a reset link has been sent."}

    reset_token          = secrets.token_urlsafe(32)
    u.reset_token        = reset_token
    u.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
    db.commit()

    reset_url = f"{APP_BASE_URL}/reset-password?token={reset_token}"
    send_email(
        to       = u.email,
        subject  = "🔑 Reset your AI Career Assistant password",
        html_body= f"""
        <div style="font-family:Arial,sans-serif;max-width:500px;margin:auto;padding:30px">
          <h2 style="color:#4F46E5">Reset Your Password</h2>
          <p>Hi <b>{u.name}</b>, we received a request to reset your password.</p>
          <a href="{reset_url}" style="background:#4F46E5;color:#fff;padding:12px 24px;
             border-radius:6px;text-decoration:none;display:inline-block;margin:16px 0">
            Reset Password
          </a>
          <p style="color:#888;font-size:12px">This link expires in <b>1 hour</b>. If you didn't request this, ignore this email.</p>
        </div>"""
    )
    return {"message": "If that email exists, a reset link has been sent."}


@app.post("/auth/reset-password", tags=["Auth"])
def reset_password(req: ResetPasswordReq, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.reset_token == req.token).first()
    if not u:
        raise HTTPException(400, "Invalid or expired reset token")
    if not u.reset_token_expiry or datetime.utcnow() > u.reset_token_expiry:
        raise HTTPException(400, "Reset token has expired. Please request a new one.")
    u.hashed_pw          = hash_pw(req.new_password)
    u.reset_token        = None
    u.reset_token_expiry = None
    db.commit()
    send_email(
        to       = u.email,
        subject  = "✅ Password reset successful",
        html_body= f"""
        <div style="font-family:Arial,sans-serif;max-width:500px;margin:auto;padding:30px">
          <h2 style="color:#4F46E5">Password Reset Successful</h2>
          <p>Hi <b>{u.name}</b>, your password has been reset successfully.</p>
          <p>You can now log in with your new password.</p>
        </div>"""
    )
    return {"message": "✅ Password reset successful. You can now log in."}


# ── Token refresh ─────────────────────────────────────────────────────

@app.post("/auth/refresh", tags=["Auth"])
def refresh_token(u=Depends(require_user)):
    return {"access_token": make_token(u.id), "token_type": "bearer"}


class PromoteReq(BaseModel):
    email:        str
    admin_secret: str


@app.post("/auth/promote", tags=["Auth"])
def promote_admin(req: PromoteReq, db: Session = Depends(get_db)):
    if not ADMIN_SECRET or req.admin_secret != ADMIN_SECRET:
        raise HTTPException(403, "Invalid admin secret")
    u = db.query(User).filter(User.email == req.email.lower()).first()
    if not u:
        raise HTTPException(404, "User not found")
    u.is_admin = 1; db.commit()
    return {"promoted": True, "user": u.email}


# ── Admin dependency ──────────────────────────────────────────────────
def require_admin(u=Depends(get_current_user)):
    if not u or u.is_admin != 1:
        raise HTTPException(403, "Admin access required")
    return u


# ════════════════════════════════════════════════════════════════════
#  CRUD — USERS  (admin only)
# ════════════════════════════════════════════════════════════════════

class UserUpdateReq(BaseModel):
    name:  Optional[str] = None
    email: Optional[str] = None


@app.get("/admin/users", tags=["Admin"])
def list_users(skip: int = 0, limit: int = 50, _=Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).offset(skip).limit(limit).all()
    return [{"id": u.id, "name": u.name, "email": u.email, "is_admin": u.is_admin, "created_at": u.created_at} for u in users]


@app.get("/admin/users/{user_id}", tags=["Admin"])
def get_user(user_id: int, _=Depends(require_admin), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id == user_id).first()
    if not u: raise HTTPException(404, "User not found")
    return {"id": u.id, "name": u.name, "email": u.email, "created_at": u.created_at}


@app.patch("/admin/users/{user_id}", tags=["Admin"])
def update_user(user_id: int, req: UserUpdateReq, _=Depends(require_admin), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id == user_id).first()
    if not u: raise HTTPException(404, "User not found")
    if req.name:  u.name = req.name.strip()
    if req.email:
        if db.query(User).filter(User.email == req.email.lower(), User.id != user_id).first():
            raise HTTPException(400, "Email already in use")
        u.email = req.email.strip().lower()
    db.commit(); db.refresh(u)
    return {"id": u.id, "name": u.name, "email": u.email, "updated": True}


@app.delete("/admin/users/{user_id}", tags=["Admin"])
def delete_user(user_id: int, _=Depends(require_admin), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id == user_id).first()
    if not u: raise HTTPException(404, "User not found")
    ud = db.query(UserData).filter(UserData.user_id == user_id).first()
    if ud: db.delete(ud)
    db.delete(u); db.commit()
    uid = str(user_id)
    for store in [RESUME_STORE, PROFILE_STORE]: store.pop(uid, None)
    for cache in [ANALYZE_CACHE, ROADMAP_CACHE]:
        for k in [k for k in cache if k[0] == uid]: del cache[k]
    return {"deleted": True, "user_id": user_id}


# ════════════════════════════════════════════════════════════════════
#  CRUD — USER DATA  (admin only)
# ════════════════════════════════════════════════════════════════════

class UserDataUpdateReq(BaseModel):
    resume_text:   Optional[str] = None
    profile_json:  Optional[str] = None
    analyses_json: Optional[str] = None
    roadmaps_json: Optional[str] = None
    chat_json:     Optional[str] = None


@app.get("/admin/user-data", tags=["Admin"])
def list_user_data(skip: int = 0, limit: int = 50, _=Depends(require_admin), db: Session = Depends(get_db)):
    rows = db.query(UserData).offset(skip).limit(limit).all()
    return [{"id": r.id, "user_id": r.user_id, "has_resume": bool(r.resume_text),
             "analyses_count": len(json.loads(r.analyses_json or "{}")),
             "roadmaps_count": len(json.loads(r.roadmaps_json or "{}")),
             "chat_count": len(json.loads(r.chat_json or "[]")),
             "updated_at": r.updated_at} for r in rows]


@app.get("/admin/user-data/{user_id}", tags=["Admin"])
def get_user_data(user_id: int, _=Depends(require_admin), db: Session = Depends(get_db)):
    ud = db.query(UserData).filter(UserData.user_id == user_id).first()
    if not ud: raise HTTPException(404, "No data found for this user")
    return {"id": ud.id, "user_id": ud.user_id, "resume_text": ud.resume_text,
            "profile": json.loads(ud.profile_json or "{}"),
            "analyses": json.loads(ud.analyses_json or "{}"),
            "roadmaps": json.loads(ud.roadmaps_json or "{}"),
            "chat_history": json.loads(ud.chat_json or "[]"),
            "updated_at": ud.updated_at}


@app.patch("/admin/user-data/{user_id}", tags=["Admin"])
def update_user_data(user_id: int, req: UserDataUpdateReq, _=Depends(require_admin), db: Session = Depends(get_db)):
    ud = db.query(UserData).filter(UserData.user_id == user_id).first()
    if not ud: raise HTTPException(404, "No data found")
    if req.resume_text   is not None: ud.resume_text   = req.resume_text
    if req.profile_json  is not None: ud.profile_json  = req.profile_json
    if req.analyses_json is not None: ud.analyses_json = req.analyses_json
    if req.roadmaps_json is not None: ud.roadmaps_json = req.roadmaps_json
    if req.chat_json     is not None: ud.chat_json     = req.chat_json
    save_ud(db, ud)
    return {"updated": True, "user_id": user_id}


@app.delete("/admin/user-data/{user_id}", tags=["Admin"])
def delete_user_data(user_id: int, _=Depends(require_admin), db: Session = Depends(get_db)):
    ud = db.query(UserData).filter(UserData.user_id == user_id).first()
    if not ud: raise HTTPException(404, "No data found")
    ud.resume_text = ""; ud.profile_json = "{}"; ud.analyses_json = "{}"
    ud.roadmaps_json = "{}"; ud.chat_json = "[]"
    save_ud(db, ud)
    uid = str(user_id)
    for store in [RESUME_STORE, PROFILE_STORE]: store.pop(uid, None)
    for cache in [ANALYZE_CACHE, ROADMAP_CACHE]:
        for k in [k for k in cache if k[0] == uid]: del cache[k]
    return {"cleared": True, "user_id": user_id}


@app.delete("/admin/user-data/{user_id}/chat", tags=["Admin"])
def clear_chat(user_id: int, _=Depends(require_admin), db: Session = Depends(get_db)):
    ud = db.query(UserData).filter(UserData.user_id == user_id).first()
    if not ud: raise HTTPException(404, "No data found")
    ud.chat_json = "[]"; save_ud(db, ud)
    return {"cleared": True, "field": "chat_history"}


@app.delete("/admin/user-data/{user_id}/analyses", tags=["Admin"])
def clear_analyses(user_id: int, _=Depends(require_admin), db: Session = Depends(get_db)):
    ud = db.query(UserData).filter(UserData.user_id == user_id).first()
    if not ud: raise HTTPException(404, "No data found")
    ud.analyses_json = "{}"; save_ud(db, ud)
    return {"cleared": True, "field": "analyses"}


# ════════════════════════════════════════════════════════════════════
#  ADMIN EXTRA ENDPOINTS
# ════════════════════════════════════════════════════════════════════

@app.get("/admin/stats", tags=["Admin"])
def admin_stats(_=Depends(require_admin), db: Session = Depends(get_db)):
    """Summary stats for the admin dashboard."""
    total_users    = db.query(User).count()
    verified_users = db.query(User).filter(User.is_verified == 1).count()
    admin_users    = db.query(User).filter(User.is_admin == 1).count()
    total_data     = db.query(UserData).count()
    resumes        = db.query(UserData).filter(UserData.resume_text != "").count()
    recent_users   = db.query(User).order_by(User.created_at.desc()).limit(5).all()
    return {
        "total_users":    total_users,
        "verified_users": verified_users,
        "unverified":     total_users - verified_users,
        "admin_users":    admin_users,
        "total_data":     total_data,
        "resumes_uploaded": resumes,
        "recent_users": [
            {"id": u.id, "name": u.name, "email": u.email,
             "is_admin": bool(u.is_admin), "is_verified": bool(u.is_verified),
             "created_at": str(u.created_at)} for u in recent_users
        ],
    }


@app.post("/admin/users/{user_id}/promote", tags=["Admin"])
def admin_promote(user_id: int, _=Depends(require_admin), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id == user_id).first()
    if not u: raise HTTPException(404, "User not found")
    u.is_admin = 1; db.commit()
    return {"promoted": True, "user_id": user_id}


@app.post("/admin/users/{user_id}/demote", tags=["Admin"])
def admin_demote(user_id: int, _=Depends(require_admin), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.id == user_id).first()
    if not u: raise HTTPException(404, "User not found")
    u.is_admin = 0; db.commit()
    return {"demoted": True, "user_id": user_id}


# ════════════════════════════════════════════════════════════════════
#  ADMIN MANAGEMENT  (admin only)
# ════════════════════════════════════════════════════════════════════

class AdminAddByEmailReq(BaseModel):
    email: str


class AdminCreateReq(BaseModel):
    name:     str
    email:    str
    password: str

    @validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v): raise ValueError("Need uppercase letter")
        if not re.search(r"[a-z]", v): raise ValueError("Need lowercase letter")
        if not re.search(r"\d",    v): raise ValueError("Need a number")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v): raise ValueError("Need special character")
        return v

    @validator("email")
    def email_format(cls, v):
        v = v.strip().lower()
        if not re.match(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$", v):
            raise ValueError("Invalid email format")
        return v


class AdminDemoteByEmailReq(BaseModel):
    email: str


class AdminResetPasswordReq(BaseModel):
    email:        str
    new_password: str

    @validator("new_password")
    def pw_strength(cls, v):
        if len(v) < 8: raise ValueError("Minimum 8 characters")
        if not re.search(r"[A-Z]", v): raise ValueError("Need uppercase letter")
        if not re.search(r"[a-z]", v): raise ValueError("Need lowercase letter")
        if not re.search(r"\d",    v): raise ValueError("Need a number")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v): raise ValueError("Need special character")
        return v


class BulkPromoteReq(BaseModel):
    emails: List[str]


# ── 1. Promote existing user to admin by email ────────────────────────
@app.post("/admin/admins/promote-by-email", tags=["Admin Management"])
def promote_by_email(req: AdminAddByEmailReq, admin=Depends(require_admin), db: Session = Depends(get_db)):
    """Promote an existing registered user to admin by their email address."""
    email = req.email.strip().lower()
    u = db.query(User).filter(User.email == email).first()
    if not u:
        raise HTTPException(404, f"No user found with email: {email}. They must register first.")
    if u.id == admin.id:
        raise HTTPException(400, "You are already an admin.")
    if u.is_admin == 1:
        return {"message": f"{email} is already an admin.", "already_admin": True}
    u.is_admin = 1; db.commit()
    return {"promoted": True, "email": email, "user_id": u.id, "name": u.name}


# ── 2. Create a brand-new admin account directly ──────────────────────
@app.post("/admin/admins/create", tags=["Admin Management"])
def create_admin_account(req: AdminCreateReq, _=Depends(require_admin), db: Session = Depends(get_db)):
    """Create a new user account that is immediately an admin (no OTP needed)."""
    email = req.email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, f"Email {email} is already registered.")
    u = User(
        name        = req.name.strip(),
        email       = email,
        hashed_pw   = hash_pw(req.password),
        is_admin    = 1,
        is_verified = 1,
    )
    db.add(u); db.commit(); db.refresh(u)
    get_or_create_ud(db, u.id)
    return {
        "created": True,
        "message": f"Admin account created for {email}",
        "user": {"id": u.id, "name": u.name, "email": u.email, "is_admin": True}
    }


# ── 3. List all current admins ────────────────────────────────────────
@app.get("/admin/admins", tags=["Admin Management"])
def list_admins(_=Depends(require_admin), db: Session = Depends(get_db)):
    """Get a list of all admin users."""
    admins = db.query(User).filter(User.is_admin == 1).all()
    return {
        "total_admins": len(admins),
        "admins": [
            {"id": u.id, "name": u.name, "email": u.email,
             "is_verified": bool(u.is_verified), "created_at": str(u.created_at)}
            for u in admins
        ]
    }


# ── 4. Demote admin by email ──────────────────────────────────────────
@app.post("/admin/admins/demote-by-email", tags=["Admin Management"])
def demote_by_email(req: AdminDemoteByEmailReq, admin=Depends(require_admin), db: Session = Depends(get_db)):
    """Remove admin privileges from a user by their email address."""
    email = req.email.strip().lower()
    u = db.query(User).filter(User.email == email).first()
    if not u:
        raise HTTPException(404, f"No user found with email: {email}")
    if u.id == admin.id:
        raise HTTPException(400, "You cannot demote yourself.")
    if u.is_admin != 1:
        return {"message": f"{email} is not an admin.", "already_normal": True}
    u.is_admin = 0; db.commit()
    return {"demoted": True, "email": email, "user_id": u.id}


# ── 5. Bulk promote multiple users to admin ───────────────────────────
@app.post("/admin/admins/bulk-promote", tags=["Admin Management"])
def bulk_promote(req: BulkPromoteReq, _=Depends(require_admin), db: Session = Depends(get_db)):
    """Promote multiple users to admin at once by providing a list of emails."""
    results = []
    for email in req.emails:
        email = email.strip().lower()
        u = db.query(User).filter(User.email == email).first()
        if not u:
            results.append({"email": email, "status": "not_found"})
        elif u.is_admin == 1:
            results.append({"email": email, "status": "already_admin"})
        else:
            u.is_admin = 1
            results.append({"email": email, "status": "promoted", "user_id": u.id})
    db.commit()
    promoted = [r for r in results if r["status"] == "promoted"]
    return {
        "total_requested": len(req.emails),
        "promoted":        len(promoted),
        "results":         results
    }


# ── 6. Reset any user's password (admin override) ─────────────────────
@app.post("/admin/admins/reset-password", tags=["Admin Management"])
def admin_reset_user_password(req: AdminResetPasswordReq, _=Depends(require_admin), db: Session = Depends(get_db)):
    """Force-reset a user's password by email. Does not require the old password."""
    email = req.email.strip().lower()
    u = db.query(User).filter(User.email == email).first()
    if not u:
        raise HTTPException(404, f"No user found with email: {email}")
    u.hashed_pw = hash_pw(req.new_password)
    db.commit()
    return {"reset": True, "email": email, "message": f"Password for {email} has been reset successfully."}


# ── 7. Check if a specific email is admin ─────────────────────────────
@app.get("/admin/admins/check", tags=["Admin Management"])
def check_is_admin(email: str = Query(...), _=Depends(require_admin), db: Session = Depends(get_db)):
    """Check whether a specific email address has admin privileges."""
    u = db.query(User).filter(User.email == email.strip().lower()).first()
    if not u:
        raise HTTPException(404, f"No user found with email: {email}")
    return {
        "email":    u.email,
        "name":     u.name,
        "is_admin": bool(u.is_admin),
        "user_id":  u.id
    }


# ════════════════════════════════════════════════════════════════════
#  USER DASHBOARD
# ════════════════════════════════════════════════════════════════════

@app.get("/user/dashboard", tags=["User"])
def user_dashboard(u=Depends(require_user), db: Session = Depends(get_db)):
    ud       = get_or_create_ud(db, u.id)
    analyses = json.loads(ud.analyses_json or "{}")
    roadmaps = json.loads(ud.roadmaps_json or "{}")
    return {
        "user":           {"id": u.id, "name": u.name, "email": u.email,
                           "member_since": u.created_at.strftime("%b %Y")},
        "has_resume":     bool(ud.resume_text),
        "profile":        json.loads(ud.profile_json or "{}"),
        "analyses":       analyses,
        "roadmaps":       roadmaps,
        "analyses_count": len(analyses),
        "roadmaps_count": len(roadmaps),
        "chat_history":   json.loads(ud.chat_json or "[]"),
        "last_updated":   ud.updated_at.isoformat() if ud.updated_at else None,
    }


# ════════════════════════════════════════════════════════════════════
#  UPLOAD
# ════════════════════════════════════════════════════════════════════

@app.post("/upload", tags=["Resume"])
async def upload_resume(file: UploadFile = File(...),
                        u=Depends(get_current_user), db: Session = Depends(get_db)):
    uid = _uid(u)
    if not file.filename.endswith('.pdf'):
        return {"error": "Only PDF files are supported"}
    temp = f"temp_{uid}_{file.filename}"
    try:
        with open(temp, "wb") as f: shutil.copyfileobj(file.file, f)
        text = extract_text(temp)
        os.remove(temp)
        if not text or len(text.strip()) < 50:
            return {"error": "Resume appears empty or unreadable"}
        profile = parse_profile(text)
        RESUME_STORE[uid]  = text
        PROFILE_STORE[uid] = profile
        for k in [k for k in ANALYZE_CACHE if k[0] == uid]: del ANALYZE_CACHE[k]
        for k in [k for k in ROADMAP_CACHE if k[0] == uid]: del ROADMAP_CACHE[k]
        if u:
            ud = get_or_create_ud(db, u.id)
            ud.resume_text   = text
            ud.profile_json  = json.dumps(profile)
            ud.analyses_json = "{}"
            ud.roadmaps_json = "{}"
            save_ud(db, ud)
        return {"message": "Resume uploaded successfully", "text_length": len(text), "profile": profile}
    except Exception as e:
        if os.path.exists(temp): os.remove(temp)
        return {"error": f"Failed: {str(e)}"}


@app.get("/extract-profile", tags=["Resume"])
def extract_profile(u=Depends(get_current_user), db: Session = Depends(get_db)):
    uid = _uid(u)
    if uid in PROFILE_STORE: return PROFILE_STORE[uid]
    if u:
        ud = get_or_create_ud(db, u.id)
        if ud.profile_json and ud.profile_json != "{}":
            p = json.loads(ud.profile_json)
            RESUME_STORE[uid]  = ud.resume_text or ""
            PROFILE_STORE[uid] = p
            return p
    return {"error": "Upload resume first"}


# ════════════════════════════════════════════════════════════════════
#  ANALYZE  — Agent 1 enriches skill gap data
# ════════════════════════════════════════════════════════════════════

@app.get("/analyze", tags=["Analyze"])
async def analyze(role: str = Query(...),
                  u=Depends(get_current_user), db: Session = Depends(get_db)):
    uid = _uid(u)
    if uid not in RESUME_STORE and u:
        ud = get_or_create_ud(db, u.id)
        if ud.resume_text: RESUME_STORE[uid] = ud.resume_text
    if uid not in RESUME_STORE:
        return {"error": "Please upload a resume first"}

    rk = role.strip().lower(); ck = (uid, rk)
    if ck in ANALYZE_CACHE: return ANALYZE_CACHE[ck]
    if u:
        ud = get_or_create_ud(db, u.id)
        a  = json.loads(ud.analyses_json or "{}")
        if rk in a: ANALYZE_CACHE[ck] = a[rk]; return a[rk]

    text   = RESUME_STORE[uid]
    prompt = f"""You are a career advisor. Analyse this resume for: {role}

RESUME:
{text[:1200]}

Reply in EXACTLY this format:

MATCH SCORE: [number]/100
[One sentence why]

TOP 5 STRENGTHS:
1. **[Skill]**: [why good for {role}]
   • Why it matters: [reason]
   • How to leverage: [action]
2. **[Skill]**: [why good]
   • Why it matters: [reason]
   • How to leverage: [action]
3. **[Skill]**: [why good]
   • Why it matters: [reason]
   • How to leverage: [action]
4. **[Skill]**: [why good]
   • Why it matters: [reason]
   • How to leverage: [action]
5. **[Skill]**: [why good]
   • Why it matters: [reason]
   • How to leverage: [action]

TOP 5 MISSING SKILLS:
1. **[Skill]**: [why needed for {role}]
   • Learn: [free resource]
   • Timeline: [X weeks]
2. **[Skill]**: [why needed]
   • Learn: [resource]
   • Timeline: [X weeks]
3. **[Skill]**: [why needed]
   • Learn: [resource]
   • Timeline: [X weeks]
4. **[Skill]**: [why needed]
   • Learn: [resource]
   • Timeline: [X weeks]
5. **[Skill]**: [why needed]
   • Learn: [resource]
   • Timeline: [X weeks]

OVERALL SUMMARY:
[Two sentences: current fit + top priority action]

MARKET_JSON:
{{"avg_salary":"X-Y LPA","demand":80,"openings":"X,000+","growth":"+X%","top_companies":["C1","C2","C3"]}}

PROJECTS_JSON:
[{{"title":"Project name for {role}","description":"One compelling sentence.","tech":["T1","T2","T3"],"icon":"emoji","difficulty":"Beginner","duration":"2-3 weeks","steps":["Step 1: setup","Step 2: build core feature","Step 3: add data/API","Step 4: deploy"]}},{{"title":"Intermediate project","description":"One compelling sentence.","tech":["T1","T2"],"icon":"emoji","difficulty":"Intermediate","duration":"3-5 weeks","steps":["Step 1: design architecture","Step 2: implement","Step 3: add auth/tests","Step 4: deploy to cloud"]}},{{"title":"Advanced capstone","description":"One compelling sentence.","tech":["T1","T2","T3"],"icon":"emoji","difficulty":"Intermediate","duration":"4-6 weeks","steps":["Step 1: plan","Step 2: build MVP","Step 3: add advanced features","Step 4: showcase"]}}]
CRITICAL: ALL projects must be SPECIFICALLY for a {role} professional. No generic CRUD apps.

SKILLS_DATA:
{{"have":[{{"name":"SkillName","level":75}},{{"name":"SkillName","level":60}},{{"name":"SkillName","level":85}},{{"name":"SkillName","level":70}},{{"name":"SkillName","level":65}}],"missing":[{{"name":"SkillName","level":0,"target":80}},{{"name":"SkillName","level":0,"target":75}},{{"name":"SkillName","level":0,"target":70}},{{"name":"SkillName","level":0,"target":85}},{{"name":"SkillName","level":0,"target":75}}]}}"""

    raw           = await ask_gemini_async(prompt)
    score         = extract_score(raw)
    analysis_text = raw.split("MARKET_JSON:")[0].strip() if "MARKET_JSON:" in raw else raw

    # ── Market ────────────────────────────────────────────────────────
    market = {"avg_salary":"10-20 LPA","demand":78,"openings":"10,000+","growth":"+18%",
              "top_companies":["Google","Microsoft","Infosys"]}
    _m = extract_json_block(raw, "MARKET_JSON", "{")
    if _m:
        try: market = json.loads(_m)
        except Exception as e: print(f"[Analyze] market parse error: {e}")

    # ── Projects — use bracket-balanced extractor, then role-specific fallback ──
    projects = get_role_projects(role)
    _p = extract_json_block(raw, "PROJECTS_JSON", "[")
    if _p:
        try:
            p = json.loads(_p)
            if isinstance(p, list) and len(p) >= 2:
                projects = p
                print(f"[Analyze] ✅ Gemini projects parsed: {len(p)}")
            else:
                print(f"[Analyze] ⚠️ Gemini projects list too short, using role fallback")
        except Exception as e:
            print(f"[Analyze] ⚠️ projects parse error: {e} — using role fallback")
    else:
        print(f"[Analyze] ⚠️ PROJECTS_JSON not found in Gemini output — using role fallback")

    # ── Skills data ───────────────────────────────────────────────────
    skills_data = {"have": [], "missing": []}
    _s = extract_json_block(raw, "SKILLS_DATA", "{")
    if _s:
        try: skills_data = json.loads(_s)
        except Exception as e: print(f"[Analyze] skills_data parse error: {e}")

    result = {"analysis": analysis_text, "match_score": score, "market": market,
              "projects": projects, "skills_data": skills_data,
              "role": role, "analyzed_at": datetime.utcnow().isoformat()}

    # ── Agent 1: enrich with real job-data skill gap ──────────────────
    try:
        agent_gaps = await run_skill_gap_agent_async(text, role)
        if agent_gaps.get("have") or agent_gaps.get("missing_critical"):
            result["agent_skill_gap"] = agent_gaps
            # Back-fill skills_data if Gemini left it empty
            if not skills_data.get("have") and agent_gaps.get("have"):
                skills_data["have"] = [{"name": s, "level": 70} for s in agent_gaps["have"][:5]]
            if not skills_data.get("missing") and agent_gaps.get("missing_critical"):
                skills_data["missing"] = [{"name": s, "level": 0, "target": 75}
                                           for s in agent_gaps["missing_critical"][:5]]
            result["skills_data"] = skills_data
    except Exception as e:
        print(f"[Analyze] Agent enrichment error: {e}")

    # ── Confidence (heuristic, transparent) ──────────────────────────
    try:
        confidence = compute_confidence(text, role, result)
        try:
            rs = score_resume(text)
            confidence["resume_score"]  = rs["resume_score"]
            confidence["resume_grade"]  = rs["resume_grade"]
            confidence["resume_reason"] = rs["reason"]
        except Exception as e:
            print(f"[Analyze] Resume score error: {e}")
        result["confidence"] = confidence
    except Exception as e:
        print(f"[Analyze] Confidence error: {e}")
        result["confidence"] = None

    ANALYZE_CACHE[ck] = result
    if u:
        ud = get_or_create_ud(db, u.id)
        a  = json.loads(ud.analyses_json or "{}"); a[rk] = result
        ud.analyses_json = json.dumps(a); save_ud(db, ud)

    print(f"[Analyze] ✅ Done. match_score={score}, agent_gap={'yes' if result.get('agent_skill_gap') else 'no'}")
    return result


# ════════════════════════════════════════════════════════════════════
#  ANALYZE STREAM  — sends partial results via SSE as they arrive
# ════════════════════════════════════════════════════════════════════

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/analyze/stream", tags=["Analyze"])
async def analyze_stream(role: str = Query(...),
                         token: str = Query(None),
                         db: Session = Depends(get_db)):
    """SSE — streams skill gap results in chunks as they arrive.
    Events: status | score | agent_gap | market | projects | skills_data | analysis | confidence | done | error"""

    u = None
    if token and token not in REVOKED_TOKENS:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            uid_str = payload.get("sub")
            if uid_str:
                u = db.query(User).filter(User.id == int(uid_str)).first()
        except Exception:
            pass

    uid = _uid(u); rk = role.strip().lower(); ck = (uid, rk)

    async def generate():
        if uid not in RESUME_STORE and u:
            ud = get_or_create_ud(db, u.id)
            if ud.resume_text: RESUME_STORE[uid] = ud.resume_text
        if uid not in RESUME_STORE:
            yield _sse("error", {"message": "Please upload a resume first"}); return

        text = RESUME_STORE[uid]

        # Return cache instantly
        if ck in ANALYZE_CACHE:
            c = ANALYZE_CACHE[ck]
            yield _sse("score",       {"match_score": c.get("match_score"), "role": role})
            yield _sse("market",      c.get("market", {}))
            yield _sse("projects",    {"projects": c.get("projects", [])})
            yield _sse("skills_data", c.get("skills_data", {}))
            yield _sse("analysis",    {"text": c.get("analysis", "")})
            if c.get("agent_skill_gap"): yield _sse("agent_gap", c["agent_skill_gap"])
            if c.get("confidence"):      yield _sse("confidence", c["confidence"])
            yield _sse("done", {"cached": True}); return

        yield _sse("status", {"message": f"🔍 Analysing your profile for {role}…"})

        prompt = f"""You are a career advisor. Analyse this resume for: {role}

RESUME:
{text[:1200]}

Reply in EXACTLY this format:

MATCH SCORE: [number]/100
[One sentence why]

TOP 5 STRENGTHS:
1. **[Skill]**: [why good for {role}]
   • Why it matters: [reason]
   • How to leverage: [action]
2. **[Skill]**: [why good]
   • Why it matters: [reason]
   • How to leverage: [action]
3. **[Skill]**: [why good]
   • Why it matters: [reason]
   • How to leverage: [action]
4. **[Skill]**: [why good]
   • Why it matters: [reason]
   • How to leverage: [action]
5. **[Skill]**: [why good]
   • Why it matters: [reason]
   • How to leverage: [action]

TOP 5 MISSING SKILLS:
1. **[Skill]**: [why needed for {role}]
   • Learn: [free resource]
   • Timeline: [X weeks]
2. **[Skill]**: [why needed]
   • Learn: [resource]
   • Timeline: [X weeks]
3. **[Skill]**: [why needed]
   • Learn: [resource]
   • Timeline: [X weeks]
4. **[Skill]**: [why needed]
   • Learn: [resource]
   • Timeline: [X weeks]
5. **[Skill]**: [why needed]
   • Learn: [resource]
   • Timeline: [X weeks]

OVERALL SUMMARY:
[Two sentences: current fit + top priority action]

MARKET_JSON:
{{"avg_salary":"X-Y LPA","demand":80,"openings":"X,000+","growth":"+X%","top_companies":["C1","C2","C3"]}}

PROJECTS_JSON:
[{{"title":"Project name for {role}","description":"One compelling sentence.","tech":["T1","T2","T3"],"icon":"emoji","difficulty":"Beginner","duration":"2-3 weeks","steps":["Step 1: setup","Step 2: build core feature","Step 3: add data/API","Step 4: deploy"]}},{{"title":"Intermediate project","description":"One compelling sentence.","tech":["T1","T2"],"icon":"emoji","difficulty":"Intermediate","duration":"3-5 weeks","steps":["Step 1: design architecture","Step 2: implement","Step 3: add auth/tests","Step 4: deploy to cloud"]}},{{"title":"Advanced capstone","description":"One compelling sentence.","tech":["T1","T2","T3"],"icon":"emoji","difficulty":"Intermediate","duration":"4-6 weeks","steps":["Step 1: plan","Step 2: build MVP","Step 3: add advanced features","Step 4: showcase"]}}]

SKILLS_DATA:
{{"have":[{{"name":"SkillName","level":75}},{{"name":"SkillName","level":60}},{{"name":"SkillName","level":85}},{{"name":"SkillName","level":70}},{{"name":"SkillName","level":65}}],"missing":[{{"name":"SkillName","level":0,"target":80}},{{"name":"SkillName","level":0,"target":75}},{{"name":"SkillName","level":0,"target":70}},{{"name":"SkillName","level":0,"target":85}},{{"name":"SkillName","level":0,"target":75}}]}}"""

        yield _sse("status", {"message": "⚡ Running AI analysis + live job market scan in parallel…"})

        gemini_task = asyncio.create_task(ask_gemini_async(prompt))
        agent_task  = asyncio.create_task(run_skill_gap_agent_async(text, role))
        pending     = {gemini_task, agent_task}
        agent_gaps  = {}
        gemini_raw  = ""

        while pending:
            finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                if task is agent_task:
                    try:
                        agent_gaps = task.result()
                        if isinstance(agent_gaps, dict) and (agent_gaps.get("have") or agent_gaps.get("missing_critical")):
                            yield _sse("agent_gap", {
                                "have":             agent_gaps.get("have", []),
                                "missing_critical": agent_gaps.get("missing_critical", []),
                                "missing_bonus":    agent_gaps.get("missing_bonus", []),
                                "match_percent":    agent_gaps.get("match_percent", 0),
                            })
                    except Exception as e:
                        print(f"[Stream/Analyze] Agent error: {e}"); agent_gaps = {}

                elif task is gemini_task:
                    try:
                        gemini_raw = task.result()
                    except Exception as e:
                        print(f"[Stream/Analyze] Gemini error: {e}"); gemini_raw = ""

                    score         = extract_score(gemini_raw)
                    analysis_text = gemini_raw.split("MARKET_JSON:")[0].strip() if "MARKET_JSON:" in gemini_raw else gemini_raw

                    yield _sse("score",    {"match_score": score, "role": role})
                    yield _sse("analysis", {"text": analysis_text})

                    market = {"avg_salary":"10-20 LPA","demand":78,"openings":"10,000+","growth":"+18%","top_companies":["Google","Microsoft","Infosys"]}
                    _m = extract_json_block(gemini_raw, "MARKET_JSON", "{")
                    if _m:
                        try: market = json.loads(_m)
                        except: pass
                    yield _sse("market", market)

                    projects = get_role_projects(role)
                    _p = extract_json_block(gemini_raw, "PROJECTS_JSON", "[")
                    if _p:
                        try:
                            p = json.loads(_p)
                            if isinstance(p, list) and len(p) >= 2: projects = p
                        except: pass
                    yield _sse("projects", {"projects": projects})

                    skills_data = {"have": [], "missing": []}
                    _s = extract_json_block(gemini_raw, "SKILLS_DATA", "{")
                    if _s:
                        try: skills_data = json.loads(_s)
                        except: pass
                    yield _sse("skills_data", skills_data)

        # Backfill skills_data from agent if Gemini left blanks
        if isinstance(agent_gaps, dict):
            if not skills_data.get("have") and agent_gaps.get("have"):
                skills_data["have"] = [{"name": s, "level": 70} for s in agent_gaps["have"][:5]]
            if not skills_data.get("missing") and agent_gaps.get("missing_critical"):
                skills_data["missing"] = [{"name": s, "level": 0, "target": 75} for s in agent_gaps["missing_critical"][:5]]

        result = {
            "analysis": analysis_text, "match_score": score, "market": market,
            "projects": projects, "skills_data": skills_data,
            "role": role, "analyzed_at": datetime.utcnow().isoformat(),
        }
        if isinstance(agent_gaps, dict) and (agent_gaps.get("have") or agent_gaps.get("missing_critical")):
            result["agent_skill_gap"] = agent_gaps

        try:
            confidence = compute_confidence(text, role, result)
            rs = score_resume(text)
            confidence.update({"resume_score": rs["resume_score"], "resume_grade": rs["resume_grade"], "resume_reason": rs["reason"]})
            result["confidence"] = confidence
            yield _sse("confidence", confidence)
        except Exception as e:
            print(f"[Stream/Analyze] Confidence error: {e}")

        ANALYZE_CACHE[ck] = result
        if u:
            ud = get_or_create_ud(db, u.id)
            a  = json.loads(ud.analyses_json or "{}"); a[rk] = result
            ud.analyses_json = json.dumps(a); save_ud(db, ud)

        yield _sse("done", {"match_score": score, "role": role})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ════════════════════════════════════════════════════════════════════
#  ROADMAP STREAM  — sends roadmap text + structured plan via SSE
# ════════════════════════════════════════════════════════════════════

@app.get("/roadmap/stream", tags=["Roadmap"])
async def roadmap_stream(role: str = Query(...),
                         token: str = Query(None),
                         db: Session = Depends(get_db)):
    """SSE — streams roadmap in real-time.
    Events: status | roadmap_text | structured | confidence | done | error"""

    u = None
    if token and token not in REVOKED_TOKENS:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            uid_str = payload.get("sub")
            if uid_str:
                u = db.query(User).filter(User.id == int(uid_str)).first()
        except Exception:
            pass

    uid = _uid(u); rk = role.strip().lower(); ck = (uid, rk)

    async def generate():
        if uid not in RESUME_STORE and u:
            ud = get_or_create_ud(db, u.id)
            if ud.resume_text: RESUME_STORE[uid] = ud.resume_text
        if uid not in RESUME_STORE:
            yield _sse("error", {"message": "Please upload a resume first"}); return

        text = RESUME_STORE[uid]

        if ck in ROADMAP_CACHE:
            yield _sse("roadmap_text", {"text": ROADMAP_CACHE[ck], "role": role})
            yield _sse("done", {"cached": True}); return

        yield _sse("status", {"message": f"🗺️ Building your personalised roadmap for {role}…"})

        prompt = f"""Create a 3-month learning roadmap for: {role}
Skills base: {text[:400]}

EXACT format — each item on its own line starting with "- ":

MONTH 1
📚 Skills to Learn:
- skill one
- skill two
- skill three
📖 Study Methods:
- method one
- method two
⭐ Key Topics:
- topic one
- topic two
- topic three
💼 Projects:
- project idea

MONTH 2
📚 Skills to Learn:
- skill one
- skill two
- skill three
📖 Study Methods:
- method one
- method two
⭐ Key Topics:
- topic one
- topic two
💼 Projects:
- project idea

MONTH 3
📚 Skills to Learn:
- skill one
- skill two
📖 Study Methods:
- method one
- method two
⭐ Key Topics:
- topic one
- topic two
💼 Projects:
- capstone project"""

        yield _sse("status", {"message": "⚡ Running roadmap AI + skill gap analysis simultaneously…"})

        gemini_task = asyncio.create_task(ask_gemini_async(prompt))
        agent_task  = asyncio.create_task(
            run_roadmap_agent_async(text, role, ANALYZE_CACHE.get(ck, {}).get("agent_skill_gap"))
        )
        pending      = {gemini_task, agent_task}
        gemini_text  = ""
        agent_result = {}

        while pending:
            finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                if task is gemini_task:
                    try:
                        gemini_text = task.result()
                    except Exception as e:
                        print(f"[Stream/Roadmap] Gemini error: {e}"); gemini_text = ""
                    yield _sse("roadmap_text", {"text": gemini_text, "role": role})

                elif task is agent_task:
                    try:
                        agent_result = task.result()
                    except Exception as e:
                        print(f"[Stream/Roadmap] Agent error: {e}"); agent_result = {}
                    if isinstance(agent_result, dict) and agent_result.get("months"):
                        yield _sse("structured", {
                            "months":      agent_result["months"],
                            "total_weeks": agent_result.get("total_weeks", 12),
                            "daily_hours": agent_result.get("daily_hours", 1.5),
                        })

        ROADMAP_CACHE[ck] = gemini_text
        if u:
            ud = get_or_create_ud(db, u.id)
            rm = json.loads(ud.roadmaps_json or "{}"); rm[rk] = gemini_text
            ud.roadmaps_json = json.dumps(rm); save_ud(db, ud)

        if ck in ANALYZE_CACHE and ANALYZE_CACHE[ck].get("confidence"):
            yield _sse("confidence", ANALYZE_CACHE[ck]["confidence"])

        yield _sse("done", {"role": role})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ════════════════════════════════════════════════════════════════════
#  RESUME SCORE
# ════════════════════════════════════════════════════════════════════

@app.get("/resume-score", tags=["Resume"])
async def resume_score_endpoint(u=Depends(get_current_user), db: Session = Depends(get_db)):
    uid = _uid(u)
    if uid not in RESUME_STORE and u:
        ud = get_or_create_ud(db, u.id)
        if ud.resume_text: RESUME_STORE[uid] = ud.resume_text
    if uid not in RESUME_STORE:
        return {"error": "Please upload a resume first"}
    try:
        return score_resume(RESUME_STORE[uid])
    except Exception as e:
        return {"error": f"Resume scoring failed: {str(e)}"}


# ════════════════════════════════════════════════════════════════════
#  ROADMAP  — Agent 2 builds structured month plan
# ════════════════════════════════════════════════════════════════════

@app.get("/roadmap", tags=["Roadmap"])
async def roadmap(role: str = Query(...),
                  u=Depends(get_current_user), db: Session = Depends(get_db)):
    uid = _uid(u)
    if uid not in RESUME_STORE and u:
        ud = get_or_create_ud(db, u.id)
        if ud.resume_text: RESUME_STORE[uid] = ud.resume_text
    if uid not in RESUME_STORE:
        return {"error": "Please upload a resume first"}

    rk = role.strip().lower(); ck = (uid, rk)
    if ck in ROADMAP_CACHE: return {"roadmap": ROADMAP_CACHE[ck], "role": role}
    if u:
        ud = get_or_create_ud(db, u.id)
        rm = json.loads(ud.roadmaps_json or "{}")
        if rk in rm: ROADMAP_CACHE[ck] = rm[rk]; return {"roadmap": rm[rk], "role": role}

    text   = RESUME_STORE[uid]
    prompt = f"""Create a 3-month learning roadmap for: {role}
Skills base: {text[:400]}

EXACT format — each item on its own line starting with "- ":

MONTH 1
📚 Skills to Learn:
- skill one
- skill two
- skill three
📖 Study Methods:
- method one
- method two
⭐ Key Topics:
- topic one
- topic two
- topic three
💼 Projects:
- project idea

MONTH 2
📚 Skills to Learn:
- skill one
- skill two
- skill three
📖 Study Methods:
- method one
- method two
⭐ Key Topics:
- topic one
- topic two
💼 Projects:
- project idea

MONTH 3
📚 Skills to Learn:
- skill one
- skill two
📖 Study Methods:
- method one
- method two
⭐ Key Topics:
- topic one
- topic two
💼 Projects:
- capstone project"""

    try:
        # Run Gemini text roadmap AND Agent 2 structured roadmap in parallel
        gemini_text, agent_result = await asyncio.gather(
            ask_gemini_async(prompt),
            run_roadmap_agent_async(text, role,
                                     ANALYZE_CACHE.get(ck, {}).get("agent_skill_gap")),
            return_exceptions=True,
        )

        if isinstance(gemini_text, Exception): gemini_text = ""
        if isinstance(agent_result, Exception): agent_result = {}

        ROADMAP_CACHE[ck] = gemini_text
        if u:
            ud = get_or_create_ud(db, u.id)
            rm = json.loads(ud.roadmaps_json or "{}"); rm[rk] = gemini_text
            ud.roadmaps_json = json.dumps(rm); save_ud(db, ud)

        resp = {"roadmap": gemini_text, "role": role}
        if isinstance(agent_result, dict) and agent_result.get("months"):
            resp["structured"] = agent_result   # frontend can optionally render this
        return resp

    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════════
#  MARKET / PROJECTS / JOBS
# ════════════════════════════════════════════════════════════════════

@app.get("/market-insights", tags=["Market"])
async def market_insights(role: str = Query(...), u=Depends(get_current_user)):
    uid = _uid(u); rk = role.strip().lower(); ck = (uid, rk)
    if ck in ANALYZE_CACHE: return ANALYZE_CACHE[ck].get("market", {})
    try:
        return json.loads(clean_json(await ask_gemini_async(
            f'Market for {role} India. JSON only: {{"avg_salary":"X-Y LPA","demand":80,"openings":"X,000+","growth":"+X%","top_companies":["C1","C2","C3"]}}'
        )))
    except:
        return {"avg_salary":"12-25 LPA","demand":80,"openings":"10,000+","growth":"+20%","top_companies":["Google","Microsoft","Amazon"]}


@app.get("/project-ideas", tags=["Market"])
async def project_ideas(role: str = Query(...), u=Depends(get_current_user)):
    uid = _uid(u); rk = role.strip().lower(); ck = (uid, rk)
    if ck in ANALYZE_CACHE: return {"projects": ANALYZE_CACHE[ck].get("projects", [])}
    try:
        return {"projects": json.loads(clean_json(await ask_gemini_async(
            f'3 beginner projects for {role}. JSON array only: [{{"title":"","description":"","tech":[],"icon":""}}]'
        )))}
    except:
        return {"projects": [{"title": f"{role} Dashboard","description":"Dashboard.","tech":["React","Node.js"],"icon":"📊"}]}


@app.get("/jobs", tags=["Market"])
def jobs(role: str = Query(...)):
    if not role or len(role.strip()) < 2:
        return {"jobs": [], "error": "Please provide a valid role"}
    try:
        return {"jobs": get_jobs(role)}
    except:
        return {"jobs": [], "error": "Job search failed"}


# ════════════════════════════════════════════════════════════════════
#  CHAT
# ════════════════════════════════════════════════════════════════════

class ChatMsg(BaseModel):
    role:    str
    content: str

class ChatReq(BaseModel):
    message: str
    history: Optional[List[ChatMsg]] = []

ALLOWED = ["resume","cv","ats","skill","job","career","interview","salary","learn","course",
           "roadmap","project","tech","technology","software","coding","code","programming",
           "python","java","javascript","react","node","sql","database","machine learning",
           "ml","ai","data science","cloud","aws","devops","git","github","linux","docker",
           "kubernetes","api","backend","frontend","fullstack","internship","placement",
           "campus","hire","trending","demand","package","ctc","lpa","hike","dsa","algorithm",
           "system design","college","degree","certification","study","resource","tutorial",
           "book","free course","udemy","coursera","leetcode","hackerrank","portfolio",
           "open source","startup","company","mnc","fresher","experience","gap","missing",
           "improve","tips","prepare","linkedin","advice"]


@app.post("/chat", tags=["Chat"])
async def chat(req: ChatReq, u=Depends(get_current_user), db: Session = Depends(get_db)):
    global CHAT_CALL_COUNT
    uid = _uid(u); msg = req.message.strip()
    if not msg or len(msg) < 2: return {"reply": "Please ask a clear question."}
    if not any(kw in msg.lower() for kw in ALLOWED):
        return {"reply": "🚫 I only help with **education & tech career** topics.\n\nAsk about skills, resumes, interviews, roadmaps, or salary! 😊"}

    s = get_static(msg)
    if s: return {"reply": s}

    if uid not in RESUME_STORE and u:
        ud = get_or_create_ud(db, u.id)
        if ud.resume_text: RESUME_STORE[uid] = ud.resume_text

    ml       = msg.lower()
    is_res   = any(k in ml for k in ["resume","cv","ats","missing","skill gap","my skills","lacking","improve my","my profile"])
    is_trend = any(k in ml for k in ["trending","latest","in demand","top skills","future","2025","2026"])
    is_int   = any(k in ml for k in ["interview","technical round","hr round","crack","coding round"])
    is_sal   = any(k in ml for k in ["salary","pay","ctc","package","lpa","hike"])
    is_learn = any(k in ml for k in ["learn","roadmap","how to start","course","tutorial","where to begin"])

    text = RESUME_STORE.get(uid, "")
    if is_res and not text:
        return {"reply": "📎 Please **upload your resume** first!\n\nGo to the **Dashboard** and upload your PDF."}

    res_ctx  = f"\nRESUME:\n{text[:500]}\n" if is_res and text else ""
    hist_ctx = "\nPREV:\n" + "".join(
        f"{'U' if m.role=='user' else 'B'}: {m.content[:80]}\n" for m in req.history[-2:]
    ) if req.history else ""

    if   is_res:   task = "Find 3 exact skill gaps from resume. Specific next steps."
    elif is_trend: task = "Top 5 in-demand tech skills 2025 with demand level."
    elif is_int:   task = "3 real interview questions, one prep strategy, one resource."
    elif is_sal:   task = "Salary ranges by experience. 2 negotiation tips."
    elif is_learn: task = "Step-by-step learning path. 2 free resources. Timeline."
    else:          task = "Direct practical career/tech advice."

    prompt = (
        f"CareerBot — AI career mentor. Be brief.\n{res_ctx}{hist_ctx}Task: {task}\nQ: {msg}\n\n"
        "FORMAT (max 8 lines):\n[emoji] **[Title]**\n• [point 1]\n• [point 2]\n• [point 3]\n📌 Tip: [one actionable tip]"
    )

    try:
        reply = await ask_gemini_async(prompt)
        CHAT_CALL_COUNT += 1
        if u:
            ud = get_or_create_ud(db, u.id)
            ch = json.loads(ud.chat_json or "[]")
            ch.append({"user": msg, "bot": reply, "ts": datetime.utcnow().isoformat()})
            ud.chat_json = json.dumps(ch[-50:]); save_ud(db, ud)
        return {"reply": reply}
    except Exception:
        return {"reply": "⚠️ AI unavailable. Please try again shortly."}


# ════════════════════════════════════════════════════════════════════
#  STATUS
# ════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Status"])
def root():
    """Serve the frontend app."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"status": "running", "version": "10.0"}


@app.get("/status", tags=["Status"])
def status():
    return {"status": "running", "version": "10.0"}


@app.get("/debug/stats", tags=["Status"])
def debug_stats():
    return {"chat_calls": CHAT_CALL_COUNT, "active_users": list(RESUME_STORE.keys())}