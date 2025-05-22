from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import requests
import sqlite3
import json
from datetime import datetime, timedelta
from openai import OpenAI
from contextlib import contextmanager
from typing import Generator, List, Optional
from textblob import TextBlob

# === Load environment variables ===
load_dotenv()
app = FastAPI()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_FEEDBACK_ID = os.getenv("NOTION_FEEDBACK_ID", NOTION_DATABASE_ID)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)
DB_PATH = "./chat_memory.db"
FILTERED_KEYWORDS = {"spam", "scam", "buy now", "click here"}
PERSONA_PROMPT = "You are Echo, a witty, concise assistant. Always reply informally, a bit quirky, and with practical advice."

# === Database Helpers ===
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                chat_id TEXT,
                message TEXT,
                rating TEXT,
                timestamp TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS privacy_optout (
                chat_id TEXT PRIMARY KEY,
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

def get_long_term_memory(chat_id: str, limit: int = 50):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT text FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, limit)
        )
        return [row[0] for row in cursor.fetchall()]

def store_feedback(chat_id: str, message: str, rating: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?)",
            (chat_id, message, rating, datetime.now().isoformat())
        )
        conn.commit()

def check_privacy_optout(chat_id: str) -> bool:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM privacy_optout WHERE chat_id = ?", (chat_id,))
        return cursor.fetchone() is not None

def set_privacy_optout(chat_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO privacy_optout VALUES (?, ?)",
            (chat_id, datetime.now().isoformat())
        )
        conn.commit()

def clear_privacy_optout(chat_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM privacy_optout WHERE chat_id = ?", (chat_id,))
        conn.commit()

# === Notion API Helpers ===
def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

def notion_query(database_id, filter_obj=None, sorts=None):
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    payload = {}
    if filter_obj:
        payload["filter"] = filter_obj
    if sorts:
        payload["sorts"] = sorts
    payload["page_size"] = 20
    r = requests.post(url, headers=notion_headers(), json=payload)
    return r.json().get("results", [])

def add_to_notion(title, content, notion_type="User Message", tags=None, chat_id=None, parent_id=None, date=None):
    if tags is None: tags = []
    url = "https://api.notion.com/v1/pages"
    data = {
        "parent": {"database_id": NOTION_DATABASE_ID} if not parent_id else {"page_id": parent_id},
        "properties": {
            "Title": {"title": [{"text": {"content": title}}]},
            "Type": {"select": {"name": notion_type}},
            "Tags": {"multi_select": [{"name": tag} for tag in tags]},
        },
        "children": [{
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]}
        }]
    }
    if date:
        data["properties"]["Date"] = {"date": {"start": date}}
    if chat_id:
        data["properties"]["Chat ID"] = {"rich_text": [{"text": {"content": str(chat_id)}}]}
    r = requests.post(url, headers=notion_headers(), json=data, timeout=10)
    return r.status_code in (200, 201)

def update_notion_page(page_id, properties: dict):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {"properties": properties}
    r = requests.patch(url, headers=notion_headers(), json=data, timeout=10)
    return r.status_code in (200, 201)

def archive_notion_page(page_id):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {"archived": True}
    r = requests.patch(url, headers=notion_headers(), json=data, timeout=10)
    return r.status_code == 200

# === "Real" Task/Events/Notes/Reminder Logic ===
def get_tasks(filter_date=None, status=None, tags=None, priority=None):
    filter_obj = {"and": [{"property": "Type", "select": {"equals": "Task"}}]}
    if filter_date:
        filter_obj["and"].append({"property": "Date", "date": {"equals": filter_date}})
    if status:
        filter_obj["and"].append({"property": "Status", "select": {"equals": status}})
    if tags:
        for tag in tags:
            filter_obj["and"].append({"property": "Tags", "multi_select": {"contains": tag}})
    if priority:
        filter_obj["and"].append({"property": "Priority", "select": {"equals": priority}})
    results = notion_query(NOTION_DATABASE_ID, filter_obj)
    tasks = []
    for page in results:
        title = page["properties"]["Title"]["title"][0]["text"]["content"]
        due = page["properties"].get("Date", {}).get("date", {}).get("start")
        stat = page["properties"].get("Status", {}).get("select", {}).get("name")
        prio = page["properties"].get("Priority", {}).get("select", {}).get("name")
        tasks.append({"title": title, "due": due, "status": stat, "priority": prio})
    return tasks

def list_events(date_range=None, participant=None, tags=None):
    filter_obj = {"and": [{"property": "Type", "select": {"equals": "Event"}}]}
    if date_range:
        filter_obj["and"].append({"property": "Date", "date": {"on_or_after": date_range[0], "on_or_before": date_range[1]}})
    if participant:
        filter_obj["and"].append({"property": "Participants", "multi_select": {"contains": participant}})
    if tags:
        for tag in tags:
            filter_obj["and"].append({"property": "Tags", "multi_select": {"contains": tag}})
    results = notion_query(NOTION_DATABASE_ID, filter_obj)
    events = []
    for page in results:
        title = page["properties"]["Title"]["title"][0]["text"]["content"]
        date = page["properties"].get("Date", {}).get("date", {}).get("start")
        location = page["properties"].get("Location", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
        events.append({"title": title, "date": date, "location": location})
    return events

def search_notes(keywords=None, tags=None, date_range=None):
    filter_obj = {"and": [{"property": "Type", "select": {"equals": "Note"}}]}
    if tags:
        for tag in tags:
            filter_obj["and"].append({"property": "Tags", "multi_select": {"contains": tag}})
    if date_range:
        filter_obj["and"].append({"property": "Date", "date": {"on_or_after": date_range[0], "on_or_before": date_range[1]}})
    results = notion_query(NOTION_DATABASE_ID, filter_obj)
    notes = []
    for page in results:
        title = page["properties"]["Title"]["title"][0]["text"]["content"]
        # Notion API doesn't support full text search—filter here if needed
        notes.append({"title": title})
    if keywords:
        notes = [note for note in notes if any(kw.lower() in note["title"].lower() for kw in keywords)]
    return notes

def create_task(title, due_date=None, description="", tags=None, priority=None):
    return add_to_notion(title, description, notion_type="Task", tags=tags, date=due_date)

def update_task(task_id=None, title=None, status=None, update_fields=None):
    props = {}
    if title: props["Title"] = {"title": [{"text": {"content": title}}]}
    if status: props["Status"] = {"select": {"name": status}}
    if update_fields:
        for key, val in update_fields.items():
            props[key] = {"rich_text": [{"text": {"content": val}}]}
    return update_notion_page(task_id, props)

def add_calendar_event(title, date, start_time=None, end_time=None, participants=None, description="", location="", tags=None):
    content = description
    if start_time:
        content += f"\nStart: {start_time}"
    if end_time:
        content += f"\nEnd: {end_time}"
    if location:
        content += f"\nLocation: {location}"
    if participants:
        content += f"\nParticipants: {', '.join(participants)}"
    return add_to_notion(title, content, notion_type="Event", tags=tags, date=date)

def add_note(title, content, date=None, tags=None):
    return add_to_notion(title, content, notion_type="Note", tags=tags, date=date)

def set_reminder(reminder_text, remind_at, related_task=None):
    content = f"{reminder_text}\nAt: {remind_at}"
    if related_task:
        content += f"\nTask: {related_task}"
    return add_to_notion("Reminder", content, notion_type="Reminder", date=remind_at[:10] if remind_at else None)

def cancel_reminder(reminder_id):
    return archive_notion_page(reminder_id)

def send_feedback(feedback_text, rating, related_message=None):
    content = f"{feedback_text}\nRating: {rating}"
    if related_message:
        content += f"\nRelated: {related_message}"
    return add_to_notion("Feedback", content, notion_type="Feedback", parent_id=NOTION_FEEDBACK_ID)

# === Telegram/Utility Helpers (unchanged) ===
def send_telegram_message(chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def is_filtered(text: str) -> bool:
    return any(keyword in text.lower() for keyword in FILTERED_KEYWORDS)

def analyze_sentiment(text: str) -> str:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity < -0.5:
        return "negative"
    elif polarity > 0.5:
        return "positive"
    return "neutral"

# === ADVANCED INTENT DETECTION & ROUTING ===
async def extract_intent_and_entities(text: str) -> dict:
    prompt = f"""
You are an intelligent automation assistant. Parse the following message for intent, entities, and confidence.
Available intents include:
- create_task, update_task, get_tasks, add_calendar_event, list_events, add_note, search_notes, set_reminder, cancel_reminder, send_feedback, ... (plus 100+ more; see comments for extra ideas)
User: "{text}"
Respond in valid JSON as:
{{
  "intent": "...",
  "entities": {{ ... }},
  "confidence": ...,
  "confirmation_needed": true/false
}}
If the intent is not actionable, set intent to "none".
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        result = json.loads(resp.choices[0].message.content)
    except Exception as e:
        result = {"intent": "none", "entities": {}, "confidence": 0, "confirmation_needed": True}
    return result

# === Intent Handlers (routes to real Notion logic) ===
def handle_create_task(entities, chat_id):
    title = entities.get("title", "Task")
    due_date = entities.get("due_date")
    description = entities.get("description", "")
    tags = entities.get("tags")
    priority = entities.get("priority")
    create_task(title, due_date, description, tags, priority)
    send_telegram_message(chat_id, f"✅ Task '{title}' created for {due_date or 'no date'}.")

def handle_update_task(entities, chat_id):
    task_id = entities.get("task_id")
    title = entities.get("title")
    status = entities.get("status")
    update_fields = entities.get("update_fields", {})
    update_task(task_id, title, status, update_fields)
    send_telegram_message(chat_id, f"🔄 Task updated.")

def handle_get_tasks(entities, chat_id):
    date = entities.get("filter_date")
    status = entities.get("status")
    tags = entities.get("tags")
    priority = entities.get("priority")
    tasks = get_tasks(date, status, tags, priority)
    if tasks:
        summary = "\n".join([f"• {t['title']} [{t['status'] or '-'}] ({t['due'] or '-'})" for t in tasks])
    else:
        summary = "No tasks found."
    send_telegram_message(chat_id, summary)

def handle_add_calendar_event(entities, chat_id):
    add_calendar_event(
        entities.get("title", "Event"),
        entities.get("date"),
        entities.get("start_time"),
        entities.get("end_time"),
        entities.get("participants"),
        entities.get("description"),
        entities.get("location"),
        entities.get("tags")
    )
    send_telegram_message(chat_id, "📆 Event created.")

def handle_list_events(entities, chat_id):
    date_range = entities.get("date_range")
    participant = entities.get("participant")
    tags = entities.get("tags")
    events = list_events(date_range, participant, tags)
    if events:
        summary = "\n".join([f"• {e['title']} ({e['date'] or '-'}) at {e['location'] or '-'}" for e in events])
    else:
        summary = "No events found."
    send_telegram_message(chat_id, summary)

def handle_add_note(entities, chat_id):
    add_note(
        entities.get("title", "Note"),
        entities.get("content", ""),
        entities.get("date"),
        entities.get("tags")
    )
    send_telegram_message(chat_id, "📝 Note saved!")

def handle_search_notes(entities, chat_id):
    notes = search_notes(
        entities.get("keywords"),
        entities.get("tags"),
        entities.get("date_range")
    )
    if notes:
        summary = "\n".join([f"• {n['title']}" for n in notes])
    else:
        summary = "No notes found."
    send_telegram_message(chat_id, summary)

def handle_set_reminder(entities, chat_id):
    set_reminder(
        entities.get("reminder_text"),
        entities.get("remind_at"),
        entities.get("related_task")
    )
    send_telegram_message(chat_id, "⏰ Reminder set!")

def handle_cancel_reminder(entities, chat_id):
    rid = entities.get("reminder_id")
    cancel_reminder(rid)
    send_telegram_message(chat_id, "🚫 Reminder canceled.")

def handle_send_feedback(entities, chat_id):
    send_feedback(
        entities.get("feedback_text"),
        entities.get("rating"),
        entities.get("related_message")
    )
    send_telegram_message(chat_id, "🙏 Thanks for your feedback!")

# === Intent Router Table ===
INTENT_ROUTER = {
    "create_task": handle_create_task,
    "update_task": handle_update_task,
    "get_tasks": handle_get_tasks,
    "add_calendar_event": handle_add_calendar_event,
    "list_events": handle_list_events,
    "add_note": handle_add_note,
    "search_notes": handle_search_notes,
    "set_reminder": handle_set_reminder,
    "cancel_reminder": handle_cancel_reminder,
    "send_feedback": handle_send_feedback,
    # Add more handlers as you add more functions below
}

# === Main AI Logic ===
chat_history = [{"role": "system", "content": PERSONA_PROMPT}]

async def analyze_message(message: str, context: list, long_term: list) -> str:
    try:
        chat_context = chat_history[:]
        for sender, msg in context:
            role = "assistant" if sender == "Echo" else "user"
            chat_context.append({"role": role, "content": msg})
        for long_msg in long_term:
            chat_context.append({"role": "user", "content": long_msg})
        chat_context.append({"role": "user", "content": message})
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=chat_context
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in message analysis: {e}")
        return "I encountered an error processing your message."

# === Core Endpoints (Telegram, Tasks, etc) ===
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

        if check_privacy_optout(chat_id):
            send_telegram_message(chat_id, "⚠️ Privacy Mode: Your messages are not stored. Use /forgetoff to re-enable memory.")
            return {"ok": True, "privacy_mode": True}

        if is_filtered(text):
            send_telegram_message(chat_id, "🚫 Sorry, this message was filtered for spam or prohibited keywords.")
            return {"ok": True, "filtered": True}

        # Command Handling
        if text.lower().startswith("/"):
            # (existing command logic from your code goes here...)
            pass
        else:
            intent_data = await extract_intent_and_entities(text)
            intent = intent_data.get("intent")
            entities = intent_data.get("entities", {})
            confidence = intent_data.get("confidence", 0)
            confirmation_needed = intent_data.get("confirmation_needed", False)

            context = get_recent_messages(chat_id)
            long_term = get_long_term_memory(chat_id)

            if intent in INTENT_ROUTER and confidence > 70 and not confirmation_needed:
                INTENT_ROUTER[intent](entities, chat_id)
                return {"ok": True, "intent_handled": intent}
            elif confidence < 40 or intent == "none":
                ai_response = await analyze_message(text, context, long_term)
                send_telegram_message(chat_id, f"Echo 🤖: {ai_response}\n📝 Saved to Notion.")
            else:
                send_telegram_message(chat_id, "🤖 Could you clarify your request? I need a bit more info.")

            store_message(chat_id, sender, text)
            ai_response = await analyze_message(text, context, long_term)
            add_to_notion(
                title=f"{sender} on Telegram",
                content=f"{text}\n\n---\n\n{ai_response}",
                notion_type="User Message",
                tags=["Telegram"],
                chat_id=chat_id
            )
            return {"ok": True, "intent": intent}

    except Exception as e:
        print(f"Error in webhook: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/")
def root():
    return {"status": "Echo is live 🚀"}

@app.get("/tasks-today")
def tasks_today():
    today = datetime.utcnow().date().isoformat()
    return {"tasks": get_tasks(filter_date=today)}

@app.get("/tasks-tomorrow")
def tasks_tomorrow():
    tomorrow = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
    return {"tasks": get_tasks(filter_date=tomorrow)}

@app.get("/events-this-week")
def events_this_week():
    start = datetime.utcnow().date()
    end = start + timedelta(days=7)
    return {"events": list_events(date_range=(start.isoformat(), end.isoformat()))}

@app.post("/feedback")
def feedback(payload: dict):
    chat_id = payload.get("chat_id")
    message = payload.get("message")
    rating = payload.get("rating")
    send_feedback(message, rating)
    return {"ok": True}

# --- Expandable: 100+ More Useful Smart Assistant Functions ---
# (Here is a list you can use for AI intent prompting, or as feature backlog)
"""
- add_contact, update_contact, find_contact, delete_contact
- create_project, update_project, close_project
- start_timer, stop_timer, show_timers
- generate_invoice, send_invoice, check_invoice_status
- add_expense, list_expenses, summarize_expenses
- book_flight, book_hotel, cancel_booking
- translate_text, summarize_document, extract_action_items
- scan_receipt, scan_business_card, save_document
- generate_nda, generate_contract, sign_document
- track_habit, start_workout, log_meal
- log_water, track_sleep, track_mood
- fetch_weather, set_weather_alert, get_news
- save_bookmark, open_bookmark, delete_bookmark
- create_presentation, generate_briefing, make_poll
- run_survey, show_results, send_mass_message
- share_location, find_nearby, order_taxi
- request_payment, check_balance, transfer_money
- generate_api_key, disable_api_key, check_api_usage
- schedule_zoom, join_zoom, send_zoom_invite
- remind_birthday, add_anniversary, track_gift_ideas
- list_birthdays, add_birthday_gift, archive_gift
- set_snooze, wake_me_up, create_alarm
- open_ticket, close_ticket, escalate_ticket
- start_video_call, record_call, transcribe_call
- upload_file, share_file, delete_file
- scan_for_virus, backup_files, restore_backup
- check_stock_price, add_stock_alert, buy_stock
- add_crypto_wallet, send_crypto, track_crypto_portfolio
- record_voice_note, transcribe_voice_note, send_voice_message
- start_chat, end_chat, transfer_chat
- create_reminder_for_team, send_team_update, assign_task_to_team
"""

# Continue expanding this as you wish!

# END OF FILEs