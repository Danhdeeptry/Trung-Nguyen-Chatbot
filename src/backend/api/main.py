import uuid
import time

from fastapi import FastAPI, HTTPException, Request
from requests.exceptions import HTTPError, RequestException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.backend.chatbot import run_chat_session
from history.history import ChatHistory
from src.backend.logger import setup_logger

logger = setup_logger()

app = FastAPI(
    title="Trung Nguyên ChatBot API",
    description="API for Trung Nguyên Coffee ChatBot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: str

class ChatResponse(BaseModel):
    standalone: str
    answer: str
    context: list

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"{process_time:.2f}ms"
    )
    return response

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        chat_history = ChatHistory(request.session_id)
        result = run_chat_session(
            query=request.query,
            chat_history=chat_history,
            chat_session=request.session_id
        )
        return result
    except HTTPError as e:
        logger.error(
            "Chat Endpoint HTTP Error - "
            f"User query: {request.query}"
            f"Status: {e.response.status_code} - "
            f"Response: {e.response.text} - "
            f"URL: {e.request.url}"
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        
    except Exception as e:
        logger.error(
            "Chat Endpoint Unexpected Error - "
            f"User query: {request.query}"
            f"Type: {type(e).__name__} - "
            f"Error: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/api/sessions/new")
async def create_session():
    session_id = str(uuid.uuid4())
    logger.info(f"New session created: {session_id}")
    return {"session_id": session_id}