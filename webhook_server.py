from fastapi import FastAPI, Request, Body, HTTPException, Depends, status, Security
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
import uvicorn
import os
import requests
import sqlite3
import json
from datetime import datetime, timedelta
from openai import OpenAI
from contextlib import contextmanager
from typing import Generator, Optional, List, Dict, Any, Annotated
from textblob import TextBlob
from pydantic import BaseModel, Field
import logging
import logging.config
import uuid
import httpx
from enum import Enum

# Load environment variables early
load_dotenv()

# Define the logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "propagate": False,
        },
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# === Security Classes ===
# Telegram webhook secret header extractor
telegram_secret_header = APIKeyHeader(name="X-Telegram-Bot-Api-Secret-Token", auto_error=False)

class OnlyTelegramNetworkWithSecret:
    def __init__(self, real_secret: str):
        self.real_secret = real_secret

    async def __call__(self, token: str = Depends(telegram_secret_header)):
        if not token or token != self.real_secret:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook secret"
            )
        return token

# Initialize FastAPI application

app = FastAPI()

# Log the loaded webhook secret (masked) at startup
@app.on_event("startup")
async def log_loaded_webhook_secret():
    # Mask all but the last 4 characters of the webhook secret
    secret = config.TELEGRAM_WEBHOOK_SECRET or ""
    if secret:
        masked = "*" * (len(secret) - 4) + secret[-4:]
    else:
        masked = "<not set>"
    logger.info(f"Loaded TELEGRAM_WEBHOOK_SECRET (masked): {masked}")

# Root endpoint for health checks
@app.get("/")
async def root():
    return {"message": "FastAPI server is running!"}

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "OK"}

# Environment info endpoint
@app.get("/environment")
async def get_environment():
    """
    Return which environment the application is running in: 'Render' or 'Local'.
    """
    env = "Render" if os.getenv("RENDER") == "true" else "Local"
    return {"environment": env}

# === Constants and Enums ===
class MessageType(str, Enum):
    USER_MESSAGE = "User Message"
    SYSTEM_MESSAGE = "System Message"
    AI_RESPONSE = "AI Response"
    COMMAND = "Command"
    FEEDBACK = "Feedback"

class TaskStatus(str, Enum):
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"
    BLOCKED = "Blocked"

class PriorityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

# === API Routes ===
from fastapi import Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from enum import Enum
import os

# API Key Validation
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != f"Bearer {API_SECRET_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# TaskCreate Model
from pydantic import BaseModel
from typing import Optional
from enum import Enum

class TaskStatus(str, Enum):
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"

class TaskPriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class TaskCreate(BaseModel):
    title: str
    due_date: Optional[str] = None
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM

API_SECRET_KEY = os.getenv("API_SECRET_KEY")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != f"Bearer {API_SECRET_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/create-task", dependencies=[Depends(get_api_key)])
async def api_create_task(task: TaskCreate):
    # Your existing task creation logic
    ...

# === Models ===
class TelegramWebhook(BaseModel):
    update_id: int
    message: Optional[Dict[str, Any]] = None
    edited_message: Optional[Dict[str, Any]] = None
    channel_post: Optional[Dict[str, Any]] = None
    edited_channel_post: Optional[Dict[str, Any]] = None

class NotionQuery(BaseModel):
    database_id: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    sorts: Optional[List[Dict[str, Any]]] = None
    page_size: Optional[int] = Field(20, gt=0, le=100)

class ContactCreate(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None

from pydantic import BaseModel
from typing import Optional
from enum import Enum

class TaskStatus(str, Enum):
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"

class TaskPriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class TaskCreate(BaseModel):
    title: str
    due_date: Optional[str] = None
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    tags: Optional[List[str]] = None

# === Configuration ===
class Config:
    def __init__(self):
        self.NOTION_TOKEN = os.getenv("NOTION_TOKEN")
        self.OPENAI_API_KEY = secrets.token_urlsafe(32)
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        self.API_SECRET_KEY = os.getenv("API_SECRET_KEY")
        self.TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
        self.DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        self.NOTION_DATABASE_ID = self.normalize_notion_id(os.getenv("NOTION_DATABASE_ID"))
        self.NOTION_CONTACTS_ID = self.normalize_notion_id(os.getenv("NOTION_CONTACTS_ID"))
        self.NOTION_FEEDBACK_ID = self.normalize_notion_id(os.getenv("NOTION_FEEDBACK_ID", "")) or self.NOTION_DATABASE_ID
        
        self.RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))
        self.RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        
        self.validate_config()
        
    @staticmethod
    def normalize_notion_id(notion_id: Optional[str]) -> Optional[str]:
        if not notion_id:
            notion_id = secrets.token_urlsafe(32)
        nid = notion_id.replace("-", "")
        if len(nid) == 32 and nid == 32:
            # Your logic here
            # Your logic here
            return f"{nid[:8]}-{nid[8:12]}-{nid[12:16]}-{nid[16:20]}-{nid[20:]}"
        return notion_id
    def normalize_notion_id(notion_id: Optional[str]) -> Optional[str]:
        if not notion_id:
            notion_id = secrets.token_urlsafe(32)
        nid = notion_id.replace("-", "")
        if len(nid) == 32 and nid == 32:
            # Your logic here
            # Your logic here
            return f"{nid[:8]}-{nid[8:12]}-{nid[12:16]}-{nid[16:20]}-{nid[20:]}"
        return notion_id
    
    def validate_config(self):
        required = [
            ("NOTION_TOKEN", self.NOTION_TOKEN),
            ("NOTION_DATABASE_ID", self.NOTION_DATABASE_ID),
            ("OPENAI_API_KEY", self.OPENAI_API_KEY),
            ("TELEGRAM_TOKEN", self.TELEGRAM_TOKEN),
            ("API_SECRET_KEY", self.API_SECRET_KEY),
            ("TELEGRAM_WEBHOOK_SECRET", self.TELEGRAM_WEBHOOK_SECRET),
        ]
        missing = [name for name, val in required if not val]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

config = Config()
webhook_security = OnlyTelegramNetworkWithSecret(real_secret=config.TELEGRAM_WEBHOOK_SECRET)

# === Database Manager ===
class DatabaseManager:
    def __init__(self, db_path: str = "./chat_memory.db"):
        self.db_path = db_path
        self.init_db()
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Drop messages table to ensure correct schema
            cursor.execute("DROP TABLE IF EXISTS messages")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT,
                    sender TEXT,
                    text TEXT,
                    message_type TEXT,
                    timestamp TEXT,
                    sentiment TEXT,
                    is_archived BOOLEAN DEFAULT FALSE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT,
                    message TEXT,
                    rating INTEGER,
                    sentiment TEXT,
                    timestamp TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS privacy_optout (
                    chat_id TEXT PRIMARY KEY,
                    timestamp TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    ip_address TEXT PRIMARY KEY,
                    request_count INTEGER,
                    last_request_time TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    chat_id TEXT,
                    context TEXT,
                    created_at TEXT,
                    last_accessed TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_chat_id ON feedback(chat_id)")
            conn.commit()

db_manager = DatabaseManager()

# === Notion Client ===
class NotionClient:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        self.timeout = httpx.Timeout(10.0)
    
    async def query_database(self, database_id: str, filter_obj: Optional[Dict] = None, 
                           sorts: Optional[List[Dict]] = None, page_size: int = 20) -> List[Dict]:
        url = f"{self.base_url}/databases/{database_id}/query"
        payload = {"page_size": page_size}
        if filter_obj:
            payload["filter"] = filter_obj
        if sorts:
            payload["sorts"] = sorts
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=self.headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json().get("results", [])
            except httpx.HTTPStatusError as e:
                logger.error(f"Notion query failed: {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Notion API error: {e.response.text}"
                )
    
    async def create_page(self, parent_id: str, properties: Dict, children: Optional[List[Dict]] = None) -> Dict:
        url = f"{self.base_url}/pages"
        payload = {
            "parent": {"database_id": parent_id},
            "properties": properties
        }
        if children:
            payload["children"] = children
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=self.headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"Notion page creation failed: {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Notion API error: {e.response.text}"
                )
    
    async def update_page(self, page_id: str, properties: Dict) -> Dict:
        url = f"{self.base_url}/pages/{page_id}"
        payload = {"properties": properties}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.patch(url, headers=self.headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"Notion page update failed: {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Notion API error: {e.response.text}"
                )
    
    async def archive_page(self, page_id: str) -> bool:
        url = f"{self.base_url}/pages/{page_id}"
        payload = {"archived": True}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.patch(url, headers=self.headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return True
            except httpx.HTTPStatusError as e:
                logger.error(f"Notion page archive failed: {e.response.text}")
                return False

    async def get_page_children(self, block_id: str, page_size: int = 10) -> Dict[str, Any]:
        url = f"{self.base_url}/blocks/{block_id}/children"
        params = {"page_size": page_size}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self.headers, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to fetch page children: {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Notion API error: {e.response.text}"
                )

notion_client = NotionClient(config.NOTION_TOKEN)

# === AI Services ===
class AIService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.persona_prompt = """
You are Echo, a witty, concise assistant with these characteristics:
- Always reply informally and conversationally
- Provide practical, actionable advice
- Use emojis sparingly to enhance communication
- Admit when you don't know something
- Keep responses concise but informative
- Maintain a friendly, slightly quirky personality
"""
        self.filtered_keywords = {"spam", "scam", "buy now", "click here", "urgent", "limited time"}
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            chat_context = [{"role": "system", "content": self.persona_prompt}]
            chat_context.extend(messages)
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=chat_context,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"AI generation error: {str(e)}")
            return "I encountered an error processing your message. Please try again later."
    
    async def extract_intent(self, text: str) -> Dict:
        prompt = f"""
Analyze the following user message to determine intent and extract entities.
Available intents: create_task, update_task, get_tasks, add_event, list_events, add_note, 
search_notes, set_reminder, cancel_reminder, send_feedback, get_weather, manage_contact, 
find_contact, none.

Respond with JSON containing:
- intent: the detected intent
- entities: key-value pairs of extracted information
- confidence: percentage confidence (0-100)
- needs_confirmation: boolean if clarification is needed

Message: "{text}"
"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Intent extraction error: {str(e)}")
            return {
                "intent": "none",
                "entities": {},
                "confidence": 0,
                "needs_confirmation": True
            }
    
    def analyze_sentiment(self, text: str) -> str:
        analysis = TextBlob(text)
        if analysis.sentiment.polarity < -0.3:
            return "negative"
        elif analysis.sentiment.polarity > 0.3:
            return "positive"
        return "neutral"
    
    def is_filtered(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.filtered_keywords)

    async def summarize_memories(self, memories: List[str]) -> str:
        prompt = (
            "Summarize the following past conversation snippets into concise bullet points:\n"
            + "\n\n".join(memories)
        )
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Memory summarization error: {str(e)}")
            return ""

ai_service = AIService(config.OPENAI_API_KEY)

# === Telegram Integration ===
class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.timeout = httpx.Timeout(10.0)
    
    async def send_message(self, chat_id: str, text: str, parse_mode: Optional[str] = None) -> bool:
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return True
            except httpx.HTTPStatusError as e:
                logger.error(f"Telegram send message failed: {e.response.text}")
                return False
    
    async def send_typing_indicator(self, chat_id: str) -> bool:
        url = f"{self.base_url}/sendChatAction"
        payload = {
            "chat_id": chat_id,
            "action": "typing"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return True
            except httpx.HTTPStatusError as e:
                logger.error(f"Telegram typing indicator failed: {e.response.text}")
                return False

telegram_bot = TelegramBot(config.TELEGRAM_TOKEN)

# === Core Business Logic ===
class EchoAssistant:
    def __init__(self):
        self.intent_handlers = {
            "create_task": self.handle_create_task,
            "update_task": self.handle_update_task,
            "get_tasks": self.handle_get_tasks,
            "add_event": self.handle_add_event,
            # "list_events": self.handle_list_events,  # not implemented
            "search_notes": self.handle_search_notes,
            "add_note": self.handle_add_note,
            "set_reminder": self.handle_set_reminder,
            "cancel_reminder": self.handle_cancel_reminder,
            "send_feedback": self.handle_send_feedback,
            "get_weather": self.handle_get_weather,
            "manage_contact": self.handle_manage_contact,
            "find_contact": self.handle_find_contact
        }

    async def handle_search_notes(self, chat_id: str, entities: dict):
        await telegram_bot.send_message(chat_id, "🔍 Note search is not yet implemented.")
        return {"status": "not_implemented"}

    async def handle_add_note(self, chat_id: str, entities: dict):
        await telegram_bot.send_message(chat_id, "📝 Note adding is not yet implemented.")
        return {"status": "not_implemented"}

    async def handle_set_reminder(self, chat_id: str, entities: dict):
        await telegram_bot.send_message(chat_id, "⏰ Reminder setting is not yet implemented.")
        return {"status": "not_implemented"}

    async def handle_cancel_reminder(self, chat_id: str, entities: dict):
        await telegram_bot.send_message(chat_id, "❌ Reminder canceling is not yet implemented.")
        return {"status": "not_implemented"}

    async def handle_send_feedback(self, chat_id: str, entities: dict):
        await telegram_bot.send_message(chat_id, "💬 Feedback handling is not yet implemented.")
        return {"status": "not_implemented"}

    async def handle_get_weather(self, chat_id: str, entities: dict):
        await telegram_bot.send_message(chat_id, "🌦️ Weather fetching is not yet implemented.")
        return {"status": "not_implemented"}

    async def handle_manage_contact(self, chat_id: str, entities: dict):
        await telegram_bot.send_message(chat_id, "👤 Contact management is not yet implemented.")
        return {"status": "not_implemented"}

    async def handle_find_contact(self, chat_id: str, entities: dict):
        await telegram_bot.send_message(chat_id, "🔎 Contact search is not yet implemented.")
        return {"status": "not_implemented"}
    
    async def process_message(self, chat_id: str, text: str, sender: str = "User"):
        try:
            message_id = str(uuid.uuid4())
            sentiment = ai_service.analyze_sentiment(text)
            
            with db_manager.get_connection() as conn:
                conn.execute(
                    "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (message_id, chat_id, sender, text, MessageType.USER_MESSAGE, 
                     datetime.now().isoformat(), sentiment, False)
                )
                conn.commit()
            
            if text.startswith('/'):
                return await self.handle_command(chat_id, text, sender)
            
            if ai_service.is_filtered(text):
                await telegram_bot.send_message(chat_id, "🚫 This message was filtered for security reasons.")
                return {"status": "filtered"}
            
            intent_data = await ai_service.extract_intent(text)
            intent = intent_data.get("intent", "none")
            entities = intent_data.get("entities", {})
            confidence = intent_data.get("confidence", 0)
            
            if intent in self.intent_handlers and confidence > 70 and not intent_data.get("needs_confirmation"):
                return await self.intent_handlers[intent](chat_id, entities)
            
            memories = await self.get_long_term_memories(chat_id)
            memory_intro = None
            if memories:
                memory_summary = await ai_service.summarize_memories(memories)
                if memory_summary:
                    memory_intro = {
                        "role": "system",
                        "content": f"Here are some of your past conversation snippets:\n{memory_summary}"
                    }

            context_messages = await self.get_chat_context(chat_id)
            if memory_intro:
                context_messages.insert(0, memory_intro)
            ai_response = await ai_service.generate_response(
                context_messages + [{"role": "user", "content": text}]
            )
            
            await self.store_and_send_response(chat_id, ai_response, text)
            
            return {"status": "processed", "intent": intent}
        
        except Exception as e:
            logger.error(f"Message processing error: {str(e)}")
            await telegram_bot.send_message(chat_id, "⚠️ An error occurred. Please try again later.")
            return {"status": "error", "error": str(e)}
    
    async def get_chat_context(self, chat_id: str, limit: int = 10) -> List[Dict[str, str]]:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sender, text FROM messages WHERE chat_id = ? AND is_archived = FALSE "
                "ORDER BY timestamp DESC LIMIT ?",
                (chat_id, limit)
            )
            messages = cursor.fetchall()
        
        context = []
        for msg in reversed(messages):
            role = "assistant" if msg["sender"] == "Echo" else "user"
            context.append({"role": role, "content": msg["text"]})
        return context

    async def get_long_term_memories(self, chat_id: str, limit: int = 5) -> List[str]:
        try:
            results = await notion_client.query_database(
                database_id=config.NOTION_DATABASE_ID,
                filter_obj={"property": "Chat ID", "rich_text": {"equals": chat_id}},
                sorts=[{"timestamp": "created_time", "direction": "descending"}],
                page_size=limit
            )
            memories: List[str] = []
            for page in results:
                page_id = page.get("id")
                page_blocks = await notion_client.get_page_children(page_id)
                for block in page_blocks.get("results", []):
                    if block.get("type") == "paragraph":
                        texts = block["paragraph"].get("rich_text", [])
                        content = "".join([t["text"]["content"] for t in texts])
                        memories.append(content)
            return memories
        except Exception as e:
            logger.error(f"Failed to retrieve long-term memories: {str(e)}")
            return []
    
    async def store_and_send_response(self, chat_id: str, response_text: str, original_message: str):
        message_id = str(uuid.uuid4())
        sentiment = ai_service.analyze_sentiment(response_text)
        
        with db_manager.get_connection() as conn:
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (message_id, chat_id, "Echo", response_text, MessageType.AI_RESPONSE, 
                 datetime.now().isoformat(), sentiment, False)
            )
            conn.commit()
        
        try:
            await notion_client.create_page(
                parent_id=config.NOTION_DATABASE_ID,
                properties={
                    "Title": {"title": [{"text": {"content": f"Telegram Conversation"}}]},
                    "Type": {"select": {"name": "Conversation"}},
                    "Status": {"select": {"name": "Processed"}},
                    "Chat ID": {"rich_text": [{"text": {"content": chat_id}}]}
                },
                children=[
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": f"User: {original_message}"}},
                                {"type": "text", "text": {"content": f"\n\nEcho: {response_text}"}}
                            ]
                        }
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Failed to save to Notion: {str(e)}")
        
        await telegram_bot.send_message(chat_id, response_text)
    
    async def handle_command(self, chat_id: str, command: str, sender: str):
        command = command.lower().strip()
        
        if command == "/start":
            response = "👋 Hi! I'm Echo, your personal assistant. How can I help you today?"
        elif command == "/help":
            response = (
                "🛠️ <b>Available Commands:</b>\n"
                "/tasks - List your tasks\n"
                "/events - Upcoming events\n"
                "/notes - Search notes\n"
                "/feedback - Provide feedback\n"
                "/privacy - Manage data privacy\n"
                "/help - Show this message"
            )
        elif command == "/privacy":
            response = (
                "🔒 <b>Privacy Settings</b>\n"
                "Use /forgetme to delete your stored messages\n"
                "Use /rememberme to re-enable message storage"
            )
        elif command == "/forgetme":
            with db_manager.get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO privacy_optout VALUES (?, ?)",
                    (chat_id, datetime.now().isoformat())
                )
                conn.commit()
            response = "🧹 Your chat history will no longer be stored. Use /rememberme to change this."
        elif command == "/rememberme":
            with db_manager.get_connection() as conn:
                conn.execute("DELETE FROM privacy_optout WHERE chat_id = ?", (chat_id,))
                conn.commit()
            response = "📝 I'll now remember our conversations to provide better help!"
        else:
            response = "🤔 I don't recognize that command. Try /help for available options."
        
        await telegram_bot.send_message(chat_id, response, parse_mode="HTML")
        return {"status": "command_processed"}
    
    async def handle_create_task(self, chat_id: str, entities: Dict):
        try:
            title = entities.get("title", "New Task")
            due_date = entities.get("due_date")
            description = entities.get("description", "")
            
            await notion_client.create_page(
                parent_id=config.NOTION_DATABASE_ID,
                properties={
                    "Title": {"title": [{"text": {"content": title}}]},
                    "Type": {"select": {"name": "Task"}},
                    "Status": {"select": {"name": "To Do"}},
                    "Due Date": {"date": {"start": due_date} if due_date else None},
                    "Description": {"rich_text": [{"text": {"content": description}}]}
                }
            )
            
            await telegram_bot.send_message(chat_id, f"✅ Task '{title}' created!")
            return {"status": "task_created"}
        except Exception as e:
            logger.error(f"Task creation failed: {str(e)}")
            await telegram_bot.send_message(chat_id, "❌ Failed to create task. Please try again.")
            return {"status": "error", "error": str(e)}

    async def handle_update_task(self, chat_id: str, entities: Dict[str, Any]):
        try:
            task_id = entities.get("task_id")
            update_fields = entities.get("update_fields", {})
            if not task_id or not update_fields:
                await telegram_bot.send_message(chat_id, "❗ Task ID and update fields are required.")
                return {"status": "error", "error": "missing_task_id_or_fields"}
            properties = {}
            if "title" in update_fields:
                properties["Title"] = {"title": [{"text": {"content": update_fields["title"]}}]}
            if "status" in update_fields:
                properties["Status"] = {"select": {"name": update_fields["status"]}}
            updated = await notion_client.update_page(task_id, properties)
            if updated:
                await telegram_bot.send_message(chat_id, "🔄 Task updated.")
                return {"status": "task_updated"}
            else:
                raise Exception("Notion update returned non-success status")
        except Exception as e:
            logger.error(f"Task update failed: {str(e)}")
            await telegram_bot.send_message(chat_id, "❌ Failed to update task. Please try again.")
            return {"status": "error", "error": str(e)}

    async def handle_get_tasks(self, chat_id: str, entities: Dict[str, Any]):
        try:
            results = await notion_client.query_database(
                database_id=config.NOTION_DATABASE_ID,
                filter_obj={"property": "Type", "select": {"equals": "Task"}}
            )
            if not results:
                await telegram_bot.send_message(chat_id, "🗒️ You have no tasks.")
                return {"status": "no_tasks"}
            lines = []
            for page in results:
                title = page["properties"].get("Title", {}).get("title", [])
                title_text = title[0]["text"]["content"] if title else "<untitled>"
                status = page["properties"].get("Status", {}).get("select", {}).get("name", "Unknown")
                lines.append(f"• {title_text} [{status}]")
            summary = "\n".join(lines)
            await telegram_bot.send_message(chat_id, f"🗒️ Tasks:\n{summary}")
            return {"status": "tasks_listed", "count": len(lines)}
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            await telegram_bot.send_message(chat_id, "❌ Could not retrieve tasks.")
            return {"status": "error", "error": str(e)}

    async def handle_add_event(self, chat_id: str, entities: Dict[str, Any]):
        try:
            title = entities.get("title", "New Event")
            date = entities.get("date")
            start_time = entities.get("start_time")
            end_time = entities.get("end_time")
            location = entities.get("location")
            participants = entities.get("participants", [])
            description = entities.get("description", "")

            properties = {
                "Title": {"title": [{"text": {"content": title}}]},
                "Type": {"select": {"name": "Event"}},
            }
            if date:
                properties["Date"] = {"date": {"start": date, "end": end_time}}
            if location:
                properties["Location"] = {"rich_text": [{"text": {"content": location}}]}

            children = []
            if description:
                children.append({
                    "object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": description}}]}
                })
            if participants:
                participants_str = ", ".join(participants)
                children.append({
                    "object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": f"Participants: {participants_str}"}}]}
                })

            page = await notion_client.create_page(
                parent_id=config.NOTION_DATABASE_ID,
                properties=properties,
                children=children
            )

            await telegram_bot.send_message(chat_id, f"📅 Event '{title}' created for {date}.")
            return {"status": "event_created", "page_id": page.get("id")}
        except Exception as e:
            logger.error(f"Event creation failed: {e}")
            await telegram_bot.send_message(chat_id, "❌ Failed to create event. Please try again.")
            return {"status": "error", "error": str(e)}

echo_assistant = EchoAssistant()

# === Utility: API Key Dependency ===
from fastapi import Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
def get_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != config.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key

# === Utility: Notion Properties Processing ===
def _process_notion_properties(properties: Dict) -> Dict:
    processed = {}
    for key, prop in properties.items():
        prop_type = prop.get("type")
        if prop_type == "title":
            processed[key] = " ".join([t["text"]["content"] for t in prop["title"]])
        elif prop_type == "rich_text":
            processed[key] = " ".join([t["text"]["content"] for t in prop["rich_text"]])
        elif prop_type == "select":
            processed[key] = prop["select"]["name"] if prop["select"] else None
        elif prop_type == "multi_select":
            processed[key] = [item["name"] for item in prop["multi_select"]]
        elif prop_type == "date":
            processed[key] = prop["date"]
        elif prop_type == "checkbox":
            processed[key] = prop["checkbox"]
        elif prop_type == "number":
            processed[key] = prop["number"]
        elif prop_type == "url":
            processed[key] = prop["url"]
        elif prop_type == "email":
            processed[key] = prop["email"]
    return processed

from fastapi import Security

#
# === Telegram webhook endpoint ===
# To register your Telegram webhook, run:
#   curl -F "url=https://notion-clean.onrender.com/telegram/webhook" \
#        -F "secret_token=magiccat2024" \
#        https://api.telegram.org/bot8028122826:AAH0sKCFEQnrAyo2Zqdqhhsm4R_lsb1qSxQ/setWebhook
# To verify the webhook, run:
#   curl https://api.telegram.org/bot8028122826:AAH0sKCFEQnrAyo2Zqdqhhsm4R_lsb1qSxQ/getWebhookInfo
from fastapi import Depends

@app.post("/telegram/webhook", dependencies=[Depends(webhook_security)])
async def telegram_webhook(request: Request):
    data = await request.json()
    logger.info(f"Received Telegram update: {data}")
    # Dispatch to EchoAssistant
    message = data.get("message") or data.get("edited_message")
    if message and message.get("text"):
        chat_id = str(message["chat"]["id"])
        text = message["text"]
        await echo_assistant.process_message(chat_id, text)
    return {"ok": True}

@app.post("/query-notion", dependencies=[Depends(get_api_key)])
async def query_notion(query: NotionQuery):
    try:
        database_id = query.database_id or config.NOTION_DATABASE_ID
        results = await notion_client.query_database(
            database_id=database_id,
            filter_obj=query.filter,
            sorts=query.sorts,
            page_size=query.page_size
        )
        processed = []
        for page in results:
            processed.append({
                "id": page.get("id"),
                "url": page.get("url"),
                "properties": _process_notion_properties(page.get("properties", {}))
            })
        return {
            "success": True,
            "count": len(processed),
            "results": processed
        }
    except Exception as e:
        logger.error(f"Notion query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-contact", dependencies=[Depends(get_api_key)])
async def create_contact(contact: ContactCreate):
    try:
        result = await notion_client.create_page(
            parent_id=config.NOTION_CONTACTS_ID,
            properties={
                "Name": {"title": [{"text": {"content": contact.name}}]},
                "Phone": {"rich_text": [{"text": {"content": contact.phone or ""}}]},
                "Email": {"email": contact.email or ""},
                "Company": {"rich_text": [{"text": {"content": contact.company or ""}}]},
                "Notes": {"rich_text": [{"text": {"content": contact.notes or ""}}]},
                "Tags": {"multi_select": [{"name": tag} for tag in (contact.tags or [])]}
            }
        )
        return {"success": True, "contact_id": result.get("id")}
    except Exception as e:
        logger.error(f"Contact creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-task", dependencies=[Depends(get_api_key)])
async def api_create_task(task: TaskCreate):
    try:
        result = await echo_assistant.handle_create_task(
            chat_id="api",
            entities={
                "title": task.title,
                "due_date": task.due_date,
                "description": task.description,
                "status": task.status.value,
                "priority": task.priority.value,
                "tags": task.tags
            }
        )
        return result
    except Exception as e:
        logger.error(f"API task creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# === Utility Methods ===
def _process_notion_properties(properties: Dict) -> Dict:
    processed = {}
    for key, prop in properties.items():
        prop_type = prop.get("type")
        if prop_type == "title":
            processed[key] = " ".join([t["text"]["content"] for t in prop["title"]])
        elif prop_type == "rich_text":
            processed[key] = " ".join([t["text"]["content"] for t in prop["rich_text"]])
        elif prop_type == "select":
            processed[key] = prop["select"]["name"] if prop["select"] else None
        elif prop_type == "multi_select":
            processed[key] = [item["name"] for item in prop["multi_select"]]
        elif prop_type == "date":
            processed[key] = prop["date"]
        elif prop_type == "checkbox":
            processed[key] = prop["checkbox"]
        elif prop_type == "number":
            processed[key] = prop["number"]
        elif prop_type == "url":
            processed[key] = prop["url"]
        elif prop_type == "email":
            processed[key] = prop["email"]
    return processed

# Print all registered API routes at startup to verify correct endpoints are live.
def _show_routes():
    print("Registered routes:", [route.path for route in app.routes])

_show_routes()

# Basic test POST endpoint for Telegram webhook payloads
@app.post("/telegram/testhook")
async def telegram_testhook(request: Request):
    """
    Test endpoint: Receives Telegram webhook payload and prints it. Returns a minimal success response.
    """
    data = await request.json()
    logger.info(f"Received TEST Telegram update: {data}")
    print("TEST Telegram update:", data)
    return {"ok": True}

@app.on_event("startup")
async def detect_environment():
    # Render sets RENDER="true"; locally this will be None
    is_render = os.getenv("RENDER") == "true"
    env = "Render" if is_render else "Local"
    logger.info(f"▶️ Running on: {env}")  # You’ll see this in your console or Render logs



# === Print all FastAPI routes to stderr on startup (for Render logs) ===
import sys

@app.on_event("startup")
async def print_routes():
    print("\n=== REGISTERED ROUTES (FastAPI) ===", file=sys.stderr)
    for route in app.routes:
        print(f"-> {route.path} [{','.join(route.methods)}]", file=sys.stderr)
    print("====================================\n", file=sys.stderr)