from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from conversation_manager import ConversationService, SessionMemory, new_session_memory
from llm_engine import OllamaLLM

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

llm = OllamaLLM()
service = ConversationService(llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await llm.close()


app = FastAPI(
    title="SmileCare Conversational AI",
    version="2.0.0",
    summary="Stateless local dental clinic assistant with FastAPI, WebSocket streaming, and Ollama.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class SessionPayload(BaseModel):
    session_id: Optional[str] = None
    window_size: int = 12
    history: List[Dict[str, str]] = Field(default_factory=list)
    intent: Optional[str] = None
    state: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    policy_flags: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session: Optional[SessionPayload] = None


class ChatResponse(BaseModel):
    reply: str
    intent: str
    state: str
    session: Dict[str, Any]


class SessionResponse(BaseModel):
    session: Dict[str, Any]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    llm_health = await llm.health()
    return {
        "status": "ok",
        "backend_mode": "stateless",
        "session_storage": "client-side payload",
        "llm": llm_health,
    }


@app.post("/sessions", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    return SessionResponse(session=new_session_memory().to_dict())


@app.delete("/sessions/{session_id}", response_model=SessionResponse)
async def reset_session(session_id: str) -> SessionResponse:
    session = SessionMemory.from_dict({"session_id": session_id})
    session.reset(new_session_id=False)
    return SessionResponse(session=session.to_dict())


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    session = SessionMemory.from_dict(payload.session.model_dump() if payload.session else None)
    reply = await service.reply(session, payload.message)
    return ChatResponse(reply=reply.text, intent=reply.intent, state=reply.state, session=reply.session)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "session", "session": new_session_memory().to_dict()})

    try:
        while True:
            payload = await websocket.receive_json()
            action = payload.get("action", "message")

            if action == "new_session":
                await websocket.send_json({"type": "session", "session": new_session_memory().to_dict()})
                continue

            if action == "reset":
                current = SessionMemory.from_dict(payload.get("session"))
                current.reset(new_session_id=False)
                await websocket.send_json({"type": "reset", "session": current.to_dict()})
                continue

            message = (payload.get("message") or "").strip()
            if not message:
                await websocket.send_json({"type": "error", "detail": "Message cannot be empty."})
                continue

            session = SessionMemory.from_dict(payload.get("session"))
            await websocket.send_json({
                "type": "start",
                "session": session.to_dict(),
                "intent": session.intent.value if session.intent else "OUT_OF_SCOPE",
                "state": session.state.value if session.state else "IDLE",
            })

            collected = []
            async for chunk in service.stream_reply(session, message):
                collected.append(chunk)
                await websocket.send_json({"type": "delta", "content": chunk})

            await websocket.send_json({
                "type": "end",
                "reply": "".join(collected).strip(),
                "intent": session.intent.value if session.intent else "OUT_OF_SCOPE",
                "state": session.state.value if session.state else "IDLE",
                "session": session.to_dict(),
            })
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close(code=1011)
