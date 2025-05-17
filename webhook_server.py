from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# 🔍 Debug: confirm .env values loaded
print("Using DB:", NOTION_DATABASE_ID)
print("Using Token:", NOTION_TOKEN[:8], "...")

app = FastAPI()

class TaskPayload(BaseModel):
    title: str
    notes: str = ""
    date: str = None  # Optional ISO 8601 date

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
                "title": [
                    {"text": {"content": payload.title}}
                ]
            }
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"text": {"content": payload.notes}}
                    ]
                }
            }
        ]
    }

    # Optional: add date as another paragraph
    if payload.date:
        data["children"].append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"text": {"content": f"Next Step: {payload.date}"}}
                ]
            }
        })

    res = requests.post(notion_url, headers=headers, json=data)
    return {
        "status": res.status_code,
        "response": res.json()
    }