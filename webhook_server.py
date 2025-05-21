import os
import datetime
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import openai

load_dotenv()

app = FastAPI()

# ENV
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

openai.api_key = OPENAI_API_KEY
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
}

# Store chat context in memory
chat_memory = {}

class TelegramMessage(BaseModel):
    update_id: int
    message: dict


def get_or_create_notion_page(chat_id: str, title: str) -> str:
    today = datetime.date.today().isoformat()
    query_url = "https://api.notion.com/v1/databases/{}/query".format(NOTION_DATABASE_ID)
    response = requests.post(query_url, headers={**headers, "Notion-Version": "2022-06-28"}, json={
        "filter": {
            "and": [
                {"property": "Chat ID", "rich_text": {"equals": chat_id}},
                {"property": "Date", "date": {"equals": today}}
            ]
        }
    })
    results = response.json().get("results")

    if results:
        return results[0]["id"]

    # Create a new page if not found
    create_url = "https://api.notion.com/v1/pages"
    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title": {"title": [{"text": {"content": title}}]},
            "Chat ID": {"rich_text": [{"text": {"content": chat_id}}]},
            "Date": {"date": {"start": today}}
        }
    }
    response = requests.post(create_url, headers={**headers, "Notion-Version": "2022-06-28"}, json=payload)
    return response.json().get("id")


def add_message_to_notion(page_id: str, message: str, analysis: str):
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    content = {
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": f"💬 {message}\n📊 {analysis}"}}]
                }
            }
        ]
    }
    requests.patch(url, headers={**headers, "Notion-Version": "2022-06-28"}, json=content)


def analyze_message(text: str, history: str = "") -> str:
    prompt = f"Context:\n{history}\n\nMessage:\n{text}\n\nAnalyze the message. Provide a short summary, tone (e.g. friendly, frustrated), and response recommendation."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response["choices"][0]["message"]["content"].strip()


def reply_to_telegram(chat_id: int, text: str):
    payload = {"chat_id": chat_id, "text": text}
    requests.post(TELEGRAM_API_URL, json=payload)


@app.post("/telegram")
async def telegram_webhook(update: TelegramMessage):
    message = update.message
    text = message.get("text")
    chat = message.get("chat", {})
    chat_id = str(chat.get("id"))
    sender = message.get("from", {}).get("username", "Unknown")

    # Maintain context
    if chat_id not in chat_memory:
        chat_memory[chat_id] = []
    chat_memory[chat_id].append(text)
    history = "\n".join(chat_memory[chat_id][-10:])  # Keep last 10

    # Analyze
    analysis = analyze_message(text, history)

    # Notion
    page_title = f"Chat with {sender}"
    page_id = get_or_create_notion_page(chat_id, page_title)
    add_message_to_notion(page_id, text, analysis)

    # Telegram reply
    reply_to_telegram(chat_id, "🧠 Got it! Message logged + analyzed.")

    return {"ok": True}
