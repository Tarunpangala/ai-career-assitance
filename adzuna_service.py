import requests
import os
from dotenv import load_dotenv

load_dotenv()

APP_ID  = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")


def get_jobs(role: str) -> list:
    try:
        url    = "https://api.adzuna.com/v1/api/jobs/in/search/1"
        params = {"app_id": APP_ID, "app_key": APP_KEY, "what": role, "results_per_page": 10}
        res    = requests.get(url, params=params, timeout=8)
        if res.status_code == 200:
            return res.json().get("results", [])
    except Exception as e:
        print(f"Adzuna error: {e}")
    return []
