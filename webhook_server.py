import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI

# ✅ Load environment variables first
load_dotenv()

# ✅ Access environment variables
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Now it's safe to initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Telegram API base URL
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ✅ Debug print
print("Using DB:", NOTION_DATABASE_ID)
print("Using Token:", NOTION_TOKEN[:8], "...")
print("Using Telegram:", TELEGRAM_TOKEN[:8], "...")
print("Using OpenAI:", OPENAI_API_KEY[:8], "...")

app = FastAPI()

# ✅ Task model
class TaskPayload(BaseModel):
    title: str
    notes: str = ""
    date: Optional[str] = None

# ✅ Create Notion task
@app.post("/task")
async def receive_task(payload: TaskPayload):
    print("✅ Received POST /task")
    print("Payload received:", payload.dict())

    notion_url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    data = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title": {
                "title": [{"text": {"content": payload.title}}]
            }
        },
        "children": []
    }

    if payload.notes:
        data["children"].append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"text": {"content": payload.notes}}]
            }
        })

    if payload.date:
        data["children"].append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"text": {"content": f"Next Step: {payload.date}"}}]
            }
        })

    try:
        res = requests.post(notion_url, headers=headers, json=data, timeout=10)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Notion error: {str(e)}")

    return {"status": "Task created ✅", "notion_response": res.json()}

# ✅ Telegram webhook with GPT-4 Turbo reply
@app.post("/telegram")
async def telegram_webhook(request: Request):
    body = await request.json()
    print("📩 Telegram message:", body)

    chat_id = body.get("message", {}).get("chat", {}).get("id")
    text = body.get("message", {}).get("text", "")

    if not chat_id or not text:
        return {"ok": False, "error": "Empty or invalid Telegram message payload"}

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": text}]
        )
        gpt_reply = response.choices[0].message.content
    except Exception as e:
        print("❌ GPT error:", str(e))
        gpt_reply = "⚠️ I'm having trouble thinking right now. Try again in a moment!"

    reply = {
        "chat_id": chat_id,
        "text": gpt_reply
    }

    try:
        requests.post(f"{TELEGRAM_URL}/sendMessage", json=reply, timeout=10)
    except Exception as e:
        print("❌ Telegram send error:", str(e))

    return {"ok": True}

# ✅ Root GET/HEAD route
@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    return {"message": "Echo is live 🚀"}
