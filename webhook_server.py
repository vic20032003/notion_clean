import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import redis

# ✅ Load environment variables
load_dotenv()

# ✅ Environment config
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# ✅ Clients
client = OpenAI(api_key=OPENAI_API_KEY)
redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ✅ Debug print
print("Using DB:", NOTION_DATABASE_ID)
print("Using Token:", NOTION_TOKEN[:8], "...")
print("Using Telegram:", TELEGRAM_TOKEN[:8], "...")
print("Using OpenAI:", OPENAI_API_KEY[:8], "...")

app = FastAPI()

class TaskPayload(BaseModel):
    title: str
    notes: str = ""
    date: Optional[str] = None

def store_message(chat_id, role, content):
    key = f"chat:{chat_id}"
    redis_client.rpush(key, f"{role}:{content}")
    redis_client.ltrim(key, -10, -1)

def get_history(chat_id):
    key = f"chat:{chat_id}"
    raw = redis_client.lrange(key, 0, -1)
    return [{"role": part.split(":", 1)[0], "content": part.split(":", 1)[1]} for part in raw if ":" in part]

def log_to_notion(chat_id, messages):
    try:
        conversation = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        data = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": {
                "Title": {
                    "title": [{"text": {"content": f"Chat log: {chat_id}"}}]
                }
            },
            "children": [{
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"text": {"content": conversation}}]
                }
            }]
        }
        headers = {
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        requests.post("https://api.notion.com/v1/pages", headers=headers, json=data, timeout=10)
    except Exception as e:
        print("❌ Failed to log conversation to Notion:", str(e))

@app.post("/telegram")
async def telegram_webhook(request: Request):
    body = await request.json()
    print("📩 Telegram message:", body)

    chat_id = body.get("message", {}).get("chat", {}).get("id")
    text = body.get("message", {}).get("text", "")

    if not chat_id or not text:
        return {"ok": False, "error": "Empty or invalid Telegram message payload"}

    store_message(chat_id, "user", text)
    messages = get_history(chat_id)

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages
        )
        gpt_reply = response.choices[0].message.content
    except Exception as e:
        print("❌ GPT error:", str(e))
        gpt_reply = "⚠️ I'm having trouble thinking right now. Try again in a moment!"

    store_message(chat_id, "assistant", gpt_reply)

    reply = {
        "chat_id": chat_id,
        "text": gpt_reply
    }

    try:
        requests.post(f"{TELEGRAM_URL}/sendMessage", json=reply, timeout=10)
    except Exception as e:
        print("❌ Telegram send error:", str(e))

    if text.lower().strip() == "save":
        log_to_notion(chat_id, messages)

    return {"ok": True}

@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    return {"message": "Echo is live 🚀"}
