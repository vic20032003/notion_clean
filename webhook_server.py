from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os
import requests
import sqlite3
from datetime import datetime
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# ENV variables
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# OpenAI Client (new SDK style)
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize local SQLite DB
conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        chat_id TEXT,
        sender TEXT,
        text TEXT,
        timestamp TEXT
    )
''')
conn.commit()

# Store new message
def store_message(chat_id, sender, text):
    cursor.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?)",
        (chat_id, sender, text, datetime.now().isoformat())
    )
    conn.commit()

# Retrieve recent messages (for context)
def get_recent_messages(chat_id, limit=10):
    cursor.execute(
        "SELECT sender, text FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
        (chat_id, limit)
    )
    return list(reversed(cursor.fetchall()))

# Send reply back to Telegram
def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)

# Save message + AI analysis to Notion
def add_to_notion(title, content):
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    data = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Name": {
                "title": [{"text": {"content": title}}]
            }
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "text": [{"type": "text", "text": {"content": content}}]
                }
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.status_code == 200

# Analyze current message using GPT-4o
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

# Telegram webhook entrypoint
@app.post("/telegram")
async def telegram_webhook(req: Request):
    body = await req.json()
    message = body.get("message")

    if not message:
        return {"status": "no message"}

    chat_id = message["chat"]["id"]
    sender = message["from"]["username"]
    text = message.get("text", "")

    # Store + context
    store_message(str(chat_id), sender, text)
    context = get_recent_messages(str(chat_id))

    # GPT Analysis
    ai_response = analyze_message(text, context)

    # Telegram Reply
    send_telegram_message(chat_id, f"Echo 🤖: {ai_response}\n📝 Saved to Notion.")

    # Notion Save
    add_to_notion(title=f"{sender} on Telegram", content=f"{text}\n\n---\n\n{ai_response}")

    return {"ok": True}

# Health check
@app.get("/")
def root():
    return {"status": "Echo is live 🚀"}
