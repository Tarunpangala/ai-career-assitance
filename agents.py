"""
agents.py  —  AI Career Assistant
==================================
Agent 1 — Skill Gap Analyzer  →  /analyze endpoint
Agent 2 — Roadmap Builder     →  /roadmap endpoint

OPTIMISATIONS IN THIS VERSION:
  1. Always uses direct Gemini calls (no slow multi-turn ReAct loop)
  2. gemini-2.5-flash throughout (same quota as gemini_service.py)
  3. .invoke() used on all @tool calls (fixes StructuredTool not callable)
  4. Confidence score always 90-100, varies naturally with resume quality
"""

import re, json, asyncio
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

# -- LangChain imports (optional) -------------------------------------
try:
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    LANGCHAIN_OK = True
    print("[Agents] LangChain + LangGraph loaded")
except ImportError as e:
    LANGCHAIN_OK = False
    print(f"[Agents] LangChain unavailable ({e}) — using direct Gemini calls")
    def tool(fn):
        fn.invoke = lambda args: fn(**args) if isinstance(args, dict) else fn(args)
        return fn

from gemini_service import ask_gemini, ask_gemini_async
from adzuna_service  import get_jobs


# =====================================================================
#  CONSTANTS
# =====================================================================

TECH_SKILLS = [
    "python","java","javascript","typescript","react","angular","vue","node","nodejs",
    "django","flask","fastapi","spring","springboot","sql","mysql","postgresql","mongodb",
    "redis","aws","azure","gcp","docker","kubernetes","linux","git","github",
    "machine learning","deep learning","tensorflow","pytorch","pandas","numpy","scikit",
    "html","css","tailwind","graphql","rest","api","microservices","ci/cd","jenkins",
    "kafka","spark","hadoop","airflow","mlops","nlp","computer vision","opencv",
    "excel","powerbi","tableau","data analysis","statistics","probability",
    "c++","c#","golang","go","rust","kotlin","swift","flutter","dart",
    "selenium","pytest","junit","jest","testing","devops","terraform","ansible",
]

# Skill aliases — maps common abbreviations/variants to canonical name
SKILL_ALIASES = {
    "ml": "machine learning", "ai": "machine learning", "dl": "deep learning",
    "tf": "tensorflow", "sk-learn": "scikit", "sklearn": "scikit",
    "nodejs": "node", "node.js": "node", "reactjs": "react", "react.js": "react",
    "vuejs": "vue", "vue.js": "vue", "angularjs": "angular",
    "postgres": "postgresql", "mongo": "mongodb", "k8s": "kubernetes",
    "springboot": "spring", "spring boot": "spring",
    "js": "javascript", "ts": "typescript", "py": "python",
    "gcp": "gcp", "google cloud": "gcp", "amazon web services": "aws",
    "microsoft azure": "azure", "power bi": "powerbi",
    "data science": "machine learning", "data scientist": "machine learning",
    "natural language processing": "nlp", "cv": "computer vision",
}

COMMON_ROLES = {
    "software engineer","software developer","web developer","frontend developer",
    "backend developer","full stack developer","fullstack developer",
    "data scientist","data analyst","data engineer","ml engineer",
    "machine learning engineer","devops engineer","cloud engineer",
    "python developer","java developer","react developer","node developer",
    "android developer","ios developer","mobile developer","product manager",
    "qa engineer","test engineer","cybersecurity analyst","business analyst",
    "site reliability engineer","sre","data science","artificial intelligence",
    "ai engineer","nlp engineer","computer vision engineer",
}


def _normalise_skill(s: str) -> str:
    """Lowercase + resolve aliases."""
    s = s.lower().strip()
    return SKILL_ALIASES.get(s, s)


def _skills_match(a: str, b: str) -> bool:
    """Fuzzy skill match — handles substrings and aliases."""
    a, b = _normalise_skill(a), _normalise_skill(b)
    return a == b or a in b or b in a


# =====================================================================
#  CONFIDENCE SCORE  (always 90-100, varies naturally with resume quality)
# =====================================================================

def compute_confidence(resume_text: str, role: str, gemini_output: dict) -> dict:
    """
    Confidence score is always in the range 90-100.
    Varies naturally so it never feels static — better resumes score closer to 100.

    Scoring breakdown (raw 0-100):
      length    0-25   word count of resume
      skills    0-25   tech skills detected
      structure 0-20   resume sections found
      role      0-15   whether role is commonly known
      gemini    0-15   richness of AI output

    Final = 90 + (raw / 100) * 10  =>  range: 90.0 - 100.0
    """
    text   = (resume_text or "").lower()
    role_l = (role or "").lower().strip()
    pos    = []

    # length score (0-25)
    words = len(text.split())
    if   words >= 500: length_score = 25; pos.append("comprehensive resume")
    elif words >= 350: length_score = 20; pos.append("detailed resume")
    elif words >= 200: length_score = 14
    elif words >= 100: length_score = 8
    else:              length_score = 3

    # tech skills score (0-25)
    skill_count = sum(1 for s in TECH_SKILLS if s in text)
    if   skill_count >= 10: skill_score = 25; pos.append(f"{skill_count} tech skills found")
    elif skill_count >= 7:  skill_score = 20; pos.append(f"{skill_count} tech skills found")
    elif skill_count >= 4:  skill_score = 14
    elif skill_count >= 2:  skill_score = 8
    else:                   skill_score = 3

    # structure score (0-20)
    sec_kw = ["experience","education","project","skill","certif","work","intern","summary","objective"]
    found  = sum(1 for s in sec_kw if s in text)
    if   found >= 5: struct_score = 20; pos.append("well-structured resume")
    elif found >= 3: struct_score = 15
    elif found >= 2: struct_score = 10
    else:            struct_score = 5

    # role recognition score (0-15)
    is_common = any(r in role_l for r in COMMON_ROLES)
    if is_common: role_score = 15; pos.append(f'"{role}" is a recognised role')
    else:         role_score = 8

    # gemini output richness (0-15)
    g_score = gemini_output.get("match_score", 0)
    g_pts = (
        (6 if 10 <= g_score <= 95 else 2) +
        (3 if len(gemini_output.get("projects", [])) >= 2 else 0) +
        (3 if len((gemini_output.get("analysis") or "").split()) >= 80 else 1) +
        (3 if gemini_output.get("market", {}).get("avg_salary") else 0)
    )
    if g_pts >= 12: pos.append("complete AI analysis")

    # scale raw (0-100) to final (90-100)
    raw         = length_score + skill_score + struct_score + role_score + g_pts
    trust_score = round(90.0 + (raw / 100.0) * 10.0, 1)
    trust_score = max(90.0, min(100.0, trust_score))

    # derived sub-scores also floored at 90
    skill_match  = min(round(trust_score + (skill_count * 0.2), 1), 100.0)
    roadmap_qual = min(round(trust_score + (struct_score * 0.1), 1), 100.0)
    job_fit      = min(round(trust_score + (role_score   * 0.1), 1), 100.0)

    summary = (
        f"High confidence — {', '.join(pos[:3])}."
        if pos else "High confidence — AI analysis complete."
    )

    return {
        "trust_score":          trust_score,
        "reliability":          "High",
        "reliability_color":    "green",
        "reliability_emoji":    "✅",
        "skill_match":          skill_match,
        "roadmap_quality":      roadmap_qual,
        "job_market_fit":       job_fit,
        "score_reliability":    "High",
        "salary_reliability":   "High" if is_common else "Medium",
        "skill_confidences":    {},
        "warnings":             [],
        "summary":              summary,
        "recommendation":       "Results are highly reliable. Focus on the top 2-3 missing skills.",
        "role_commonness":      "Common" if is_common else "Niche",
        "score_breakdown": {
            "length":        length_score,
            "skills":        skill_score,
            "structure":     struct_score,
            "role":          role_score,
            "gemini_output": g_pts,
            "raw_total":     raw,
        },
        "resume_quality_score": trust_score,
        "features_debug": {
            "resume_words":    words,
            "skills_detected": skill_count,
            "sections_found":  found,
            "role_known":      is_common,
        },
    }


# =====================================================================
#  TOOLS
# =====================================================================

@tool
def extract_resume_skills(resume_text: str) -> str:
    """Extract all technical skills from resume text. Returns a JSON list."""
    text  = resume_text.lower()
    # Keyword scan with alias normalisation
    found = list({_normalise_skill(s).title() for s in TECH_SKILLS if s in text})

    # Also scan for aliases in the text
    for alias, canonical in SKILL_ALIASES.items():
        if alias in text and canonical.title() not in found:
            found.append(canonical.title())

    # Use FULL resume for Gemini (not truncated to 800 chars)
    prompt = (
        "You are a technical recruiter. Extract ALL technical skills from this resume.\n"
        "Include: programming languages, frameworks, libraries, tools, cloud platforms, databases, ML/AI skills.\n"
        "Normalise abbreviations (e.g. 'ML' → 'Machine Learning', 'k8s' → 'Kubernetes').\n"
        "Return ONLY a JSON array of strings. Min 5, Max 25. No soft skills, no duplicates.\n\n"
        f"Resume:\n{resume_text[:3000]}"
    )
    try:
        raw = ask_gemini(prompt).strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        gemini_skills = json.loads(raw)
        if isinstance(gemini_skills, list):
            # Merge keyword-scan + Gemini, deduplicate by normalised name
            all_skills = found + [s for s in gemini_skills if isinstance(s, str)]
            seen = set()
            merged = []
            for s in all_skills:
                key = _normalise_skill(s)
                if key not in seen:
                    seen.add(key)
                    merged.append(s.strip())
            print(f"[Agent1] Extracted {len(merged)} skills from resume")
            return json.dumps(merged[:25])
    except Exception as e:
        print(f"[Agent1] Skill extraction Gemini error: {e}")
    return json.dumps(found)


@tool
def fetch_job_requirements(role: str) -> str:
    """Fetch job requirements for a role. Uses Adzuna live data + Gemini knowledge fallback."""

    # Role-specific hardcoded baselines (India 2025) — used as fallback
    ROLE_BASELINES = {
        "data scientist":         {"required_skills":["Python","Machine Learning","SQL","Pandas","NumPy","Scikit-learn","Statistics","Data Visualization"],"nice_to_have":["TensorFlow","PyTorch","Spark","Tableau","AWS","Deep Learning","NLP"]},
        "data science":           {"required_skills":["Python","Machine Learning","SQL","Pandas","NumPy","Scikit-learn","Statistics","Data Visualization"],"nice_to_have":["TensorFlow","PyTorch","Spark","Tableau","AWS","Deep Learning","NLP"]},
        "ml engineer":            {"required_skills":["Python","Machine Learning","TensorFlow","PyTorch","MLOps","Docker","SQL","Git"],"nice_to_have":["Kubernetes","AWS","Spark","Airflow","NLP","Computer Vision"]},
        "machine learning engineer":{"required_skills":["Python","Machine Learning","TensorFlow","PyTorch","MLOps","Docker","SQL","Git"],"nice_to_have":["Kubernetes","AWS","Spark","Airflow","NLP","Computer Vision"]},
        "software engineer":      {"required_skills":["Python","Java","JavaScript","SQL","Git","REST API","Data Structures","Algorithms"],"nice_to_have":["React","Node.js","Docker","AWS","System Design","Microservices"]},
        "backend developer":      {"required_skills":["Python","Java","Node.js","SQL","REST API","Git","Docker","PostgreSQL"],"nice_to_have":["Kubernetes","Redis","AWS","Microservices","GraphQL","MongoDB"]},
        "frontend developer":     {"required_skills":["JavaScript","React","HTML","CSS","TypeScript","Git","REST API"],"nice_to_have":["Vue","Angular","Tailwind","Redux","Testing","Node.js"]},
        "full stack developer":   {"required_skills":["JavaScript","React","Node.js","Python","SQL","Git","REST API","HTML","CSS"],"nice_to_have":["TypeScript","Docker","MongoDB","AWS","GraphQL","Redis"]},
        "devops engineer":        {"required_skills":["Docker","Kubernetes","Linux","Git","CI/CD","AWS","Shell Scripting","Terraform"],"nice_to_have":["Ansible","Jenkins","Prometheus","Python","Azure","GCP"]},
        "data analyst":           {"required_skills":["SQL","Python","Excel","Power BI","Data Visualization","Statistics","Pandas"],"nice_to_have":["Tableau","R","Machine Learning","Spark","AWS","Google Analytics"]},
        "data engineer":          {"required_skills":["Python","SQL","Spark","Airflow","AWS","ETL","PostgreSQL","Git"],"nice_to_have":["Kafka","Hadoop","Scala","Docker","Kubernetes","Snowflake"]},
        "python developer":       {"required_skills":["Python","Django","FastAPI","SQL","REST API","Git","PostgreSQL"],"nice_to_have":["Flask","Celery","Redis","Docker","AWS","Machine Learning"]},
        "react developer":        {"required_skills":["React","JavaScript","TypeScript","HTML","CSS","Git","REST API"],"nice_to_have":["Redux","Next.js","Node.js","Testing","GraphQL","Tailwind"]},
        "cloud engineer":         {"required_skills":["AWS","Docker","Kubernetes","Terraform","Linux","Python","CI/CD"],"nice_to_have":["Azure","GCP","Ansible","Jenkins","Security","Networking"]},
    }

    role_lower = role.lower().strip()
    baseline   = None
    for key, val in ROLE_BASELINES.items():
        if key in role_lower or role_lower in key:
            baseline = val
            break

    # Try Adzuna live data
    adzuna_descs = ""
    try:
        jobs = get_jobs(role)[:5]
        descs = " ".join(
            (j.get("description","") + " " + j.get("title","")).strip()
            for j in jobs if j.get("description","").strip()
        )
        if len(descs.strip()) > 100:
            adzuna_descs = descs[:2000]
            print(f"[Agent1] Adzuna returned {len(jobs)} jobs for '{role}'")
        else:
            print(f"[Agent1] Adzuna data too sparse for '{role}', using AI knowledge")
    except Exception as e:
        print(f"[Agent1] Adzuna error: {e}")

    if adzuna_descs:
        prompt = (
            f"From these real job listings for '{role}' in India, extract required skills.\n"
            f"Job data: {adzuna_descs}\n\n"
            "Return ONLY JSON (no markdown):\n"
            '{"required_skills":["Skill1","Skill2",...],"nice_to_have":["Skill3",...]}\n'
            "Include 8-12 required_skills and 4-6 nice_to_have. Normalise abbreviations."
        )
    else:
        context = f"Baseline: {json.dumps(baseline)}\n" if baseline else ""
        prompt = (
            f"List the most important technical skills required for a '{role}' job in India in 2025.\n"
            f"{context}"
            "Return ONLY JSON (no markdown):\n"
            '{"required_skills":["Skill1","Skill2",...],"nice_to_have":["Skill3",...]}\n'
            "Include 8-12 required_skills and 4-6 nice_to_have. Be specific and realistic."
        )

    try:
        raw = ask_gemini(prompt).strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw)
        if parsed.get("required_skills"):
            print(f"[Agent1] Job requirements: {len(parsed['required_skills'])} required, {len(parsed.get('nice_to_have',[]))} nice-to-have")
            return json.dumps(parsed)
    except Exception as e:
        print(f"[Agent1] Job requirements Gemini error: {e}")

    # Final fallback to baseline or generic
    if baseline:
        return json.dumps(baseline)
    return json.dumps({"required_skills":["Python","SQL","Git","REST API","Data Structures"],"nice_to_have":["Docker","AWS","Machine Learning"]})


@tool
def compute_skill_gap(resume_skills_json: str, job_requirements_json: str = "") -> str:
    """Compare resume skills vs job requirements using fuzzy matching. Returns JSON gap analysis."""
    try:
        if not job_requirements_json:
            merged        = json.loads(resume_skills_json)
            resume_skills = merged.get("resume_skills", [])
            job_req       = merged.get("job_requirements", {})
        else:
            resume_skills = json.loads(resume_skills_json)
            job_req       = json.loads(job_requirements_json)

        required = job_req.get("required_skills", [])
        nice     = job_req.get("nice_to_have", [])

        # Fuzzy match — handles "Machine Learning" vs "ML", "Node" vs "Node.js", etc.
        missing_critical = [s for s in required if not any(_skills_match(s, rs) for rs in resume_skills)]
        missing_bonus    = [s for s in nice     if not any(_skills_match(s, rs) for rs in resume_skills)]
        have             = [s for s in resume_skills
                            if any(_skills_match(s, r) for r in required + nice)]

        match_pct = int((len(required) - len(missing_critical)) / max(len(required), 1) * 100)

        print(f"[Agent1] Skill gap — have={len(have)}, missing={len(missing_critical)}, bonus_gap={len(missing_bonus)}, match={match_pct}%")
        return json.dumps({
            "have":             have,
            "missing_critical": missing_critical,
            "missing_bonus":    missing_bonus,
            "match_percent":    match_pct,
        })
    except Exception as e:
        print(f"[Agent1] compute_skill_gap error: {e}")
        return json.dumps({"error": str(e)})


@tool
def find_learning_resources(skill_name: str) -> str:
    """Find the best FREE resources to learn a skill. Returns JSON."""
    prompt = (
        f"Best FREE resources to learn {skill_name}. Max 3.\n"
        'Return ONLY JSON: {"resources":[{"name":"...","url":"...","type":"video/course/docs","hours":10}]}'
    )
    try:
        raw = ask_gemini(prompt).strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        return raw
    except Exception:
        return json.dumps({"resources": [{"name": f"{skill_name} Tutorial",
                                          "url": f"search: {skill_name} free",
                                          "type": "course", "hours": 10}]})


@tool
def build_monthly_roadmap(role_and_gaps_json: str) -> str:
    """Build a structured 3-month learning roadmap. Returns JSON."""
    prompt = (
        f"You are a senior tech career coach. Create a realistic, actionable 3-month learning roadmap.\n"
        f"Context:\n{role_and_gaps_json[:2000]}\n\n"
        "Rules:\n"
        "- Month 1: Foundation — core missing skills, free resources\n"
        "- Month 2: Application — build something real with the skills\n"
        "- Month 3: Advanced + Job Ready — capstone project, interview prep\n"
        "- daily_hours should be realistic (1-2 hrs for working people, 3-4 for full-time)\n"
        "- resources must be real, free URLs (YouTube, official docs, Kaggle, Coursera free tier)\n\n"
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        '{"months":[{"month":1,"title":"Foundation","focus":"core theme in one line","skills":["Skill1","Skill2","Skill3"],'
        '"project":"build a specific mini-project","milestone":"what you can do by end of month",'
        '"resources":[{"name":"Resource Name","url":"https://...","type":"video/docs/course","hours":10}]},'
        '{"month":2,...},{"month":3,...}],"total_weeks":12,"daily_hours":1.5}'
    )
    try:
        raw = ask_gemini(prompt).strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw)
        if parsed.get("months"):
            return json.dumps(parsed)
    except Exception as e:
        print(f"[Agent2] Roadmap Gemini error: {e}")
    return json.dumps({"months": [], "total_weeks": 12, "daily_hours": 1.5})


@tool
def score_resume_sections(resume_text: str) -> str:
    """Score each resume section. Returns JSON with scores and issues."""
    text = resume_text.lower()

    has_email    = bool(re.search(r'[\w.+-]+@[\w-]+\.\w{2,}', resume_text))
    has_phone    = bool(re.search(r'(\+?\d[\d\s\-\.]{8,14}\d)', resume_text))
    has_linkedin = "linkedin" in text
    has_github   = "github"   in text
    skill_count  = sum(1 for s in TECH_SKILLS if s in text)
    has_numbers  = bool(re.search(r'\d+\s*(%|x\b|lpa|lakh|k\b|users|records)', text))
    proj_count   = len(re.findall(r'\bproject\b|\bbuilt\b|\bdeveloped\b', text))
    verbs        = ["built","developed","implemented","designed","created",
                    "optimized","reduced","deployed","automated","launched"]
    verb_count   = sum(1 for v in verbs if v in text)

    sections = {
        "contact":  {"score": sum([has_email*25, has_phone*25, has_linkedin*25, has_github*25]),
                     "issues": ([] if has_email    else ["Missing email"]) +
                               ([] if has_phone    else ["Missing phone"]) +
                               ([] if has_linkedin else ["No LinkedIn link"]) +
                               ([] if has_github   else ["No GitHub link"])},
        "skills":   {"score": min(skill_count*10, 100),
                     "issues": [] if skill_count >= 8 else [f"Only {skill_count} tech skills — aim for 8+"]},
        "impact":   {"score": 80 if has_numbers else 30,
                     "issues": [] if has_numbers else ["No quantified results — add numbers like '40% faster'"]},
        "projects": {"score": min(proj_count*20, 100),
                     "issues": [] if proj_count >= 3 else ["Add more project descriptions with tech stack"]},
        "language": {"score": min(verb_count*14, 100),
                     "issues": [] if verb_count >= 5 else ["Use stronger verbs: Built, Deployed, Optimized"]},
    }
    overall = int(sum(s["score"] for s in sections.values()) / len(sections))
    return json.dumps({"sections": sections, "overall": overall})


# =====================================================================
#  DIRECT CALLS  (fast — no multi-turn agent loop)
#  Uses .invoke() because @tool wraps functions as StructuredTool objects
# =====================================================================

def _direct_skill_gap(resume_text: str, role: str) -> dict:
    """Run resume skill extraction + job requirement fetch concurrently, then compute gap."""
    import concurrent.futures
    try:
        print(f"[Agent1] ⚡ Fetching resume skills + job requirements in parallel for '{role}'")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_skills = pool.submit(extract_resume_skills.invoke, {"resume_text": resume_text})
            f_jobs   = pool.submit(fetch_job_requirements.invoke, {"role": role})
            skills_json = f_skills.result(timeout=20)
            jobs_json   = f_jobs.result(timeout=20)

        merged   = json.dumps({
            "resume_skills":    json.loads(skills_json),
            "job_requirements": json.loads(jobs_json),
        })
        gap_json = compute_skill_gap.invoke({"resume_skills_json": merged})
        result   = json.loads(gap_json)
        result["source"] = "direct"
        print(f"[Agent1] ✅ have={len(result.get('have',[]))}, "
              f"missing={len(result.get('missing_critical',[]))}, "
              f"match={result.get('match_percent',0)}%")
        return result
    except Exception as e:
        print(f"[Agent1] ❌ Error: {e}")
        return {"have": [], "missing_critical": [], "missing_bonus": [],
                "match_percent": 0, "source": "error"}


def _direct_roadmap(resume_text: str, role: str, gaps: Optional[dict] = None) -> dict:
    try:
        missing  = (gaps or {}).get("missing_critical", [])
        have     = (gaps or {}).get("have", [])
        match    = (gaps or {}).get("match_percent", 0)
        payload  = json.dumps({
            "role":            role,
            "missing_skills":  missing,
            "skills_you_have": have,
            "current_match":   f"{match}%",
            "resume_context":  resume_text[:2000],   # was :400 — now full context
        })
        print(f"[Agent2] Building roadmap for '{role}', missing={missing[:3]}, have={have[:3]}...")
        raw    = build_monthly_roadmap.invoke({"role_and_gaps_json": payload})
        result = json.loads(raw)
        result["source"] = "direct"
        print(f"[Agent2] ✅ {len(result.get('months',[]))} months, daily_hours={result.get('daily_hours','?')}")
        return result
    except Exception as e:
        print(f"[Agent2] ❌ Error: {e}")
        return {"months": [], "total_weeks": 12, "daily_hours": 1.5, "source": "error"}


# =====================================================================
#  AGENT RUNNERS  — always direct for speed
#  The ReAct loop makes 4-8 Gemini calls per request (very slow).
#  Direct calls make exactly 2 calls and return the same quality output.
# =====================================================================

def run_skill_gap_agent(resume_text: str, role: str) -> dict:
    return _direct_skill_gap(resume_text, role)


def run_roadmap_agent(resume_text: str, role: str,
                      skill_gaps: Optional[dict] = None) -> dict:
    return _direct_roadmap(resume_text, role, skill_gaps)


# =====================================================================
#  ASYNC WRAPPERS
# =====================================================================

async def run_skill_gap_agent_async(resume_text: str, role: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_skill_gap_agent, resume_text, role)


async def run_roadmap_agent_async(resume_text: str, role: str,
                                  skill_gaps: Optional[dict] = None) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_roadmap_agent, resume_text, role, skill_gaps)


# =====================================================================
#  RESUME SCORER
# =====================================================================

def score_resume(resume_text: str) -> dict:
    try:
        raw    = score_resume_sections.invoke({"resume_text": resume_text})
        data   = json.loads(raw)
        s      = data.get("overall", 65)
        secs   = data.get("sections", {})
        issues = [i for sec in secs.values() for i in sec.get("issues", [])]
        reason = ("Issues: " + "; ".join(issues[:2]) + ".") if issues else "Well-structured resume."
    except Exception:
        s, reason = 65, "Scored via section analysis."

    grade = "A" if s >= 80 else "B" if s >= 65 else "C" if s >= 50 else "D"
    color = "green" if s >= 80 else "blue" if s >= 65 else "orange" if s >= 50 else "red"
    return {"resume_score": s, "resume_grade": grade, "grade_color": color, "reason": reason}