from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import sqlite3
from datetime import datetime
from openai import OpenAI

# Load environment variables
load_dotenv()

app = FastAPI()

# ENV
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize SQLite DB
conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS messages
             (chat_id TEXT, sender TEXT, text TEXT, timestamp TEXT)''')
conn.commit()

# Util: Append message to local DB
def store_message(chat_id, sender, text):
    c.execute("INSERT INTO messages VALUES (?, ?, ?, ?)", (chat_id, sender, text, datetime.now().isoformat()))
    conn.commit()

# Util: Retrieve recent messages for context
def get_recent_messages(chat_id, limit=10):
    c.execute("SELECT sender, text FROM messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?", (chat_id, limit))
    rows = c.fetchall()
    return list(reversed(rows))

# Util: Send message to Telegram
def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)

# Util: Add a new page to Notion
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
        "children": [{"object": "block", "type": "paragraph", "paragraph": {"text": [{"type": "text", "text": {"content": content}}]}}]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.status_code == 200

# Util: Analyze message with OpenAI GPT-4o
def analyze_message(message, context):
    messages = [{"role": "system", "content": "You are a helpful assistant that summarizes and understands Telegram chats."}]
    for sender, msg in context:
        messages.append({"role": "user" if sender != "Echo" else "assistant", "content": msg})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

# Telegram Webhook Route
@app.post("/telegram")
async def telegram_webhook(req: Request):
    body = await req.json()
    message = body.get("message")

    if not message:
        return {"status": "no message"}

    chat_id = message["chat"]["id"]
    sender = message["from"]["username"]
    text = message.get("text", "")

    # Store message
    store_message(str(chat_id), sender, text)

    # Get recent context
    context = get_recent_messages(str(chat_id))

    # Analyze
    ai_response = analyze_message(text, context)

    # Send reply
    send_telegram_message(chat_id, f"Echo: {ai_response}\n(Saved to Notion)")

    # Save to Notion
    add_to_notion(title=f"{sender} on Telegram", content=text + "\n---\n" + ai_response)

    return {"ok": True}

# Health check route
@app.get("/")
def root():
    return {"status": "Echo is live 🚀"}
