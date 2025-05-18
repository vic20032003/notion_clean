import os
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

# ✅ Load environment variables
load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ✅ Debug print
print("Using DB:", NOTION_DATABASE_ID)
print("Using Token:", NOTION_TOKEN[:8], "...")
print("Using Telegram:", TELEGRAM_TOKEN[:8], "...")

# ✅ FastAPI app
app = FastAPI()

# ✅ Task model for Notion
class TaskPayload(BaseModel):
    title: str
    notes: str = ""
    date: Optional[str] = None  # ISO 8601

# ✅ Notion task endpoint
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

    res = requests.post(notion_url, headers=headers, json=data)
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=res.text)

    return {"status": "Task created ✅", "notion_response": res.json()}

# ✅ Telegram webhook endpoint
@app.post("/telegram")
async def telegram_webhook(request: Request):
    body = await request.json()
    print("📩 Telegram message:", body)

    chat_id = body["message"]["chat"]["id"]
    text = body["message"]["text"]

    reply = {
        "chat_id": chat_id,
        "text": f"You said: {text}"
    }

    requests.post(f"{TELEGRAM_URL}/sendMessage", json=reply)
    return {"ok": True}

# ✅ Root GET/HEAD endpoint (health check)
@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    return {"message": "Echo is live 🚀"}
