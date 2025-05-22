from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import sqlite3
from datetime import datetime
from openai import OpenAI

# === Load environment variables from .env ===
load_dotenv()
app = FastAPI()

# === ENV variables ===
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

print("Using DB:", NOTION_DATABASE_ID)
print("Using Token:", NOTION_TOKEN[:10], "...")

# === OpenAI Client ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === SQLite DB for short-term memory setup ===
DB_PATH = "./chat_memory.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    chat_id TEXT,
    sender TEXT,
    text TEXT,
    timestamp TEXT
)
""")
conn.commit()

# === Memory Functions ===
def store_message(chat_id, sender, text):
    cursor.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?)",
        (chat_id, sender, text, datetime.now().isoformat())
    )
    conn.commit()

def get_recent_messages(chat_id, limit=10):
    cursor.execute(
        "SELECT sender, text FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
        (chat_id, limit)
    )
    return list(reversed(cursor.fetchall()))

# === Telegram Send ===
def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)

# === Notion Integration (Updated Version) ===
def add_to_notion(title, content, notion_type="User Message", tags=None, chat_id=None):
    if tags is None:
        tags = []

    now_date = datetime.now().date().isoformat()
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    data = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title": {
                "title": [{"text": {"content": title}}]
            },
            "Date": {
                "date": {"start": now_date}
            },
            "Type": {
                "select": {"name": notion_type}
            },
            "Tags": {
                "multi_select": [{"name": tag} for tag in tags]
            },
            "Chat ID": {
                "rich_text": [{"text": {"content": str(chat_id)}}]
            }
        },
        "children": [{
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": content}
                }]
            }
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        print("🔍 Notion response status:", response.status_code)
        print("🧾 Notion response text:", response.text)
    except Exception as e:
        print("❌ Exception while posting to Notion:", e)
        return False

    return response.status_code in (200, 201)

# === Echo Logs ===
LOG_DIR = "echologs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_to_file(chat_id, sender, user_message, echo_reply):
    date = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.join(LOG_DIR, f"log-{date}.txt")
    entry = f"""**Date**: {timestamp}  
**Message From Echo**: {echo_reply}  
**Related User Message**: {user_message}  
**Keywords/Themes**:  
**Sigil/Trigger Phrase (if any)**:  
---
"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(entry)

def add_log_file_to_notion(chat_id):
    date = datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(LOG_DIR, f"log-{date}.txt")
    if not os.path.exists(log_filename):
        return False
    with open(log_filename, "r", encoding="utf-8") as file:
        content = file.read()
    return add_to_notion(title=f"Log {date} from {chat_id}", content=content, notion_type="Log", tags=["Log"], chat_id=chat_id)

# === GPT Message Analysis ===
def analyze_message(message, context):
    chat_history = [{"role": "system", "content": "You are a helpful assistant that responds to Telegram messages and provides insights."}]
    for sender, msg in context:
        role = "assistant" if sender == "Echo" else "user"
        chat_history.append({"role": role, "content": msg})
    chat_history.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=chat_history
    )
    return response.choices[0].message.content.strip()

# === Telegram Webhook ===
@app.post("/telegram")
async def telegram_webhook(req: Request):
    body = await req.json()
    message = body.get("message")
    if not message:
        return {"status": "no message"}
    chat_id = message["chat"]["id"]
    sender = message["from"].get("username", "Anonymous")
    text = message.get("text", "")

    store_message(str(chat_id), sender, text)
    context = get_recent_messages(str(chat_id))
    ai_response = analyze_message(text, context)

    send_telegram_message(chat_id, f"Echo 🤖: {ai_response}\n📝 Saved to Notion.")
    add_to_notion(
        title=f"{sender} on Telegram",
        content=f"{text}\n\n---\n\n{ai_response}",
        notion_type="User Message",
        tags=["Telegram"],
        chat_id=chat_id
    )
    log_to_file(chat_id, sender, text, ai_response)

    return {"ok": True}

# === Task Endpoint ===
class TaskPayload(BaseModel):
    title: str
    notes: str
    date: str | None = None

@app.post("/task")
async def receive_task(payload: TaskPayload):
    return {"message": f"Received: {payload.title}"}

# === /test-notion with Logging ===
@app.get("/test-notion")
def test_notion():
    print("📡 /test-notion called")
    success = add_to_notion("Test Title", "This came from /test-notion route", notion_type="Test", tags=["Debug"], chat_id="test")
    print("✅ Notion Success?", success)
    return {"success": success}

# === /notion-check Endpoint ===
@app.get("/notion-check")
def notion_check():
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28"
    }
    response = requests.get(url, headers=headers)
    try:
        resp_json = response.json()
    except Exception:
        resp_json = {"error": "could not decode response"}
    return {
        "status_code": response.status_code,
        "success": response.status_code == 200,
        "response": resp_json
    }

# === Root ===
@app.get("/")
def root():
    return {"status": "Echo is live 🚀"}

__all__ = ["add_to_notion", "log_to_file", "add_log_file_to_notion"]