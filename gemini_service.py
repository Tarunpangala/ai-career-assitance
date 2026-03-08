import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL  = "gemini-2.5-flash"


def ask_gemini(prompt: str) -> str:
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        if hasattr(response, "text") and response.text:
            return response.text
        return "⚠️ No AI response received."
    except Exception as e:
        print("Gemini Error:", e)
        return "⚠️ Gemini quota exceeded. Wait 30 seconds and try again."


async def ask_gemini_async(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ask_gemini, prompt)
