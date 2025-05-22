
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import sqlite3
from datetime import datetime, timedelta
from openai import OpenAI
from contextlib import contextmanager
from typing import Generator, List

# Load environment variables
load_dotenv()

app = FastAPI()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)
DB_PATH = "./chat_memory.db"

@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
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

init_db()

def store_message(chat_id: str, sender: str, text: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?)",
            (chat_id, sender, text, datetime.now().isoformat())
        )
        conn.commit()

def get_recent_messages(chat_id: str, limit: int = 10):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sender, text FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, limit)
        )
        return list(reversed(cursor.fetchall()))

def send_telegram_message(chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def add_to_notion(title: str, content: str, notion_type: str = "User Message", tags: list = None, chat_id: str = None) -> bool:
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
        return response.status_code in (200, 201)
    except Exception as e:
        print(f"Exception while posting to Notion: {e}")
        return False

def log_to_file(chat_id: str, sender: str, user_message: str, echo_reply: str):
    try:
        os.makedirs("echologs", exist_ok=True)
        date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = os.path.join("echologs", f"log-{date}.txt")
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"Date: {timestamp}\nMessage From Echo: {echo_reply}\nRelated User Message: {user_message}\n---\n")
        return True
    except Exception as e:
        print(f"Error logging to file: {e}")
        return False

chat_history = [{
    "role": "system",
    "content": (
        "You are Echo, an intelligent assistant. Be concise, helpful, and action-oriented."
    )
}]

async def analyze_message(message: str, context: list) -> str:
    try:
        chat_context = chat_history[:]
        for sender, msg in context:
            role = "assistant" if sender == "Echo" else "user"
            chat_context.append({"role": role, "content": msg})
        chat_context.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=chat_context
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in message analysis: {e}")
        return "I encountered an error processing your message."

def summarize_tasks_for_telegram(task_list: list[str]) -> str:
    if not task_list:
        return "✅ You have no tasks scheduled for tomorrow. Enjoy your day!"

    bullet_list = "\n".join(f"- {task}" for task in task_list)
    prompt = f"Summarize the following task list in a clear, helpful Telegram message:\n\n{bullet_list}\n\nBe concise, friendly, and action-oriented."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def query_notion_database(database_id, filter=None):
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    payload = {"page_size": 10}
    if filter:
        payload["filter"] = filter

    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json().get("results", [])
    except Exception as e:
        print("❌ Failed to query Notion:", e)
        return []

@app.get("/tasks-tomorrow")
def get_tomorrow_tasks():
    tomorrow = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
    filter = {
        "and": [
            {"property": "Type", "select": {"equals": "Task"}},
            {"property": "Date", "date": {"equals": tomorrow}}
        ]
    }
    tasks = query_notion_database(NOTION_DATABASE_ID, filter)
    return {
        "date": tomorrow,
        "tasks": [
            {"title": page["properties"]["Title"]["title"][0]["text"]["content"], "id": page["id"]}
            for page in tasks
            if "Title" in page["properties"] and page["properties"]["Title"]["title"]
        ]
    }

@app.post("/telegram")
async def telegram_webhook(req: Request):
    try:
        body = await req.json()
        message = body.get("message")
        if not message:
            return {"status": "no message"}

        chat_id = str(message["chat"]["id"])
        sender = message["from"].get("username", "Anonymous")
        text = message.get("text", "")

        if "tomorrow" in text.lower() and any(k in text.lower() for k in ["task", "todo", "to-do", "schedule"]):
            tasks_data = get_tomorrow_tasks()
            titles = [task["title"] for task in tasks_data.get("tasks", [])]
            summary = summarize_tasks_for_telegram(titles)
            send_telegram_message(chat_id, f"Echo 🤖:\n{summary}\n🗓️ From Notion.")
            return JSONResponse({"ok": True, "summary": summary})

        store_message(chat_id, sender, text)
        context = get_recent_messages(chat_id)
        ai_response = await analyze_message(text, context)

        telegram_success = send_telegram_message(chat_id, f"Echo 🤖: {ai_response}\n📝 Saved to Notion.")
        notion_success = add_to_notion(
            title=f"{sender} on Telegram",
            content=f"{text}\n\n---\n\n{ai_response}",
            notion_type="User Message",
            tags=["Telegram"],
            chat_id=chat_id
        )
        log_success = log_to_file(chat_id, sender, text, ai_response)

        return {
            "ok": True,
            "telegram_sent": telegram_success,
            "notion_saved": notion_success,
            "logged": log_success
        }
    except Exception as e:
        print(f"Error in webhook: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/")
def root():
    return {"status": "Echo is live 🚀"}
