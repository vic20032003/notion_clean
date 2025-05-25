main.py:

    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.security import APIKeyHeader
    from dotenv import load_dotenv
    import os
    from typing import Optional, List
    from pydantic import BaseModel
    from enum import Enum

    # Load environment variables
    load_dotenv()

    # --- Config ---
    class Config:
        def __init__(self):
            self.API_SECRET_KEY = os.getenv("API_SECRET_KEY")
            if not self.API_SECRET_KEY:
                raise RuntimeError("Missing API_SECRET_KEY in environment")

    config = Config()

    # --- Security ---
    api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
    def get_api_key(api_key: str = Depends(api_key_header)):
        if api_key != f"Bearer {config.API_SECRET_KEY}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key"
            )
        return api_key

    # --- Models & Enums ---
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

    # --- FastAPI app ---
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "FastAPI server is running!"}

    @app.get("/health")
    async def health():
        return {"status": "OK"}

    @app.post("/create-task", dependencies=[Depends(get_api_key)])
    async def api_create_task(task: TaskCreate):
        # Dummy logic: Echo the input as successful
        return {
            "success": True,
            "title": task.title,
            "status": task.status,
            "priority": task.priority,
            "tags": task.tags
        }