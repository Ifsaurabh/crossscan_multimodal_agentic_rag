import threading
from collections import deque
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

import auth
import chat_store
import chatbot
import feedback
import llm_connection
import memory
import online_report
import quotas
import usage_tracker
from db import get_connection

app = FastAPI(title="CrossScan RAG API", version="0.2.0")

_graph = None
_graph_lock = threading.Lock()
_latencies = deque(maxlen=1000)


# ---------- models ----------

class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=256)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=chatbot.MAX_MESSAGE_CHARS)
    session_id: Optional[str] = None


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=256)
    role: str = "user"


class UpdateUserRequest(BaseModel):
    is_active: Optional[bool] = None
    role: Optional[str] = None
    daily_request_limit: Optional[int] = Field(default=None, ge=0)
    daily_token_limit: Optional[int] = Field(default=None, ge=0)
    reset_limits: bool = False


class FeedbackRequest(BaseModel):
    rating: int = Field(description="1 = helpful (thumbs up), -1 = not helpful (thumbs down)")
    comment: Optional[str] = Field(default=None, max_length=feedback.MAX_COMMENT_CHARS)


class ChatResponse(BaseModel):
    message_id: Optional[int] = None  # the answer's id, used to rate it
    session_id: Optional[str]
    answer: str
    blocked: bool
    block_reason: Optional[str]
    cache_hit: bool
    guardrail_flags: List[str]
    sources: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    latency_s: float
    is_command: bool


# ---------- dependencies ----------

def get_graph():
    """The compiled retrieval graph, built once on first use."""
    global _graph
    with _graph_lock:
        if _graph is None:
            from retrieval_graph import build_graph

            _graph = build_graph()
        return _graph


def db_conn():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def current_user(authorization: Optional[str] = Header(default=None), conn=Depends(db_conn)) -> dict:
    user = auth.get_user_by_token(conn, _bearer_token(authorization))
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return user


def admin_user(user: dict = Depends(current_user)) -> dict:
    if not auth.is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def percentile(sorted_values: list, pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(pct * (len(sorted_values) - 1))))
    return sorted_values[index]


def _quota_error(e: quotas.QuotaExceeded) -> HTTPException:
    headers = {"Retry-After": str(e.retry_after)} if e.retry_after else None
    return HTTPException(status_code=429, detail={"reason": e.reason, "message": str(e)}, headers=headers)


# ---------- public ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/login")
def login(request: LoginRequest, conn=Depends(db_conn)):
    try:
        token = auth.authenticate(conn, request.username, request.password)
    except auth.LockedOut as e:
        raise HTTPException(status_code=429, detail=str(e), headers={"Retry-After": str(e.retry_after)})
    except auth.AuthError:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"token": token, "token_type": "bearer", "expires_in_hours": auth.token_ttl_hours()}


@app.post("/auth/register", status_code=201)
def register(request: RegisterRequest, conn=Depends(db_conn)):
    """Self-service sign-up: on by default, off with ALLOW_REGISTRATION=0
    (then an admin creates accounts). Throttled service-wide per hour."""
    if not auth.registration_enabled():
        raise HTTPException(status_code=403, detail="Registration is disabled. Ask an administrator for an account.")
    try:
        quotas.check_registration_allowed()
    except quotas.QuotaExceeded as e:
        raise _quota_error(e)
    try:
        user = auth.create_user(conn, request.username, request.password, role="user")
    except auth.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"username": user["username"], "role": user["role"]}


# ---------- authenticated ----------

@app.post("/auth/logout", status_code=204)
def logout(authorization: Optional[str] = Header(default=None), conn=Depends(db_conn), user: dict = Depends(current_user)):
    token = _bearer_token(authorization)
    if token:
        auth.revoke_token(conn, token)
    return Response(status_code=204)


@app.get("/me")
def me(user: dict = Depends(current_user), conn=Depends(db_conn)):
    return {
        "username": user["username"],
        "role": user["role"],
        "quota": quotas.remaining(conn, user),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, response: Response, user: dict = Depends(current_user), conn=Depends(db_conn)):
    try:
        result = chatbot.handle_message(user, request.session_id, request.message, graph=get_graph(), conn=conn)
    except quotas.QuotaExceeded as e:
        raise _quota_error(e)
    except chatbot.ServiceUnavailable as e:
        raise HTTPException(status_code=503, detail={"reason": e.reason, "message": str(e)}, headers={"Retry-After": "60"})
    except chat_store.SessionNotFound:
        raise HTTPException(status_code=404, detail="Chat session not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    _latencies.append(result["latency_s"])
    response.headers["X-Latency-Seconds"] = str(result["latency_s"])
    return ChatResponse(**{k: v for k, v in result.items() if k in ChatResponse.model_fields})


@app.get("/chat/sessions")
def list_sessions(user: dict = Depends(current_user), conn=Depends(db_conn)):
    return chat_store.list_sessions(conn, user["user_id"])


@app.get("/chat/sessions/{session_id}")
def get_session(session_id: str, user: dict = Depends(current_user), conn=Depends(db_conn)):
    session = chat_store.get_session(conn, user["user_id"], session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {**session, "messages": chat_store.get_messages(conn, user["user_id"], session_id)}


@app.post("/chat/messages/{message_id}/feedback")
def rate_answer(message_id: int, request: FeedbackRequest, user: dict = Depends(current_user), conn=Depends(db_conn)):
    try:
        feedback.submit_feedback(conn, user["user_id"], message_id, request.rating, request.comment)
    except feedback.MessageNotFound:
        raise HTTPException(status_code=404, detail="Answer not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"message_id": message_id, "rating": request.rating}


@app.delete("/chat/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, user: dict = Depends(current_user), conn=Depends(db_conn)):
    if not chat_store.delete_session(conn, user["user_id"], session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")
    return Response(status_code=204)


@app.get("/memories")
def list_memories(user: dict = Depends(current_user), conn=Depends(db_conn)):
    return memory.list_memories(conn, user["user_id"])


@app.delete("/memories/{memory_id}", status_code=204)
def delete_memory(memory_id: int, user: dict = Depends(current_user), conn=Depends(db_conn)):
    if not memory.forget(conn, user["user_id"], memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return Response(status_code=204)


@app.delete("/memories")
def delete_all_memories(user: dict = Depends(current_user), conn=Depends(db_conn)):
    return {"deleted": memory.forget_all(conn, user["user_id"])}


# ---------- admin ----------

@app.get("/admin/users")
def admin_list_users(admin: dict = Depends(admin_user), conn=Depends(db_conn)):
    return [{**u, "quota": quotas.remaining(conn, u)} for u in auth.list_users(conn)]


@app.post("/admin/users", status_code=201)
def admin_create_user(request: CreateUserRequest, admin: dict = Depends(admin_user), conn=Depends(db_conn)):
    try:
        user = auth.create_user(conn, request.username, request.password, request.role)
    except auth.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"username": user["username"], "role": user["role"]}


@app.patch("/admin/users/{username}")
def admin_update_user(username: str, request: UpdateUserRequest, admin: dict = Depends(admin_user), conn=Depends(db_conn)):
    fields = request.model_dump(exclude={"reset_limits"}, exclude_none=True)
    if request.reset_limits:
        fields["daily_request_limit"] = None
        fields["daily_token_limit"] = None
    if username == admin["username"] and (fields.get("is_active") is False or fields.get("role") == "user"):
        raise HTTPException(status_code=400, detail="You cannot deactivate or demote your own account.")
    try:
        updated = auth.update_user(conn, username, **fields)
    except auth.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not updated:
        raise HTTPException(status_code=404, detail="User not found (or nothing to change)")
    return {"updated": username}


@app.get("/admin/online-metrics")
def admin_online_metrics(days: int = 7, admin: dict = Depends(admin_user), conn=Depends(db_conn)):
    """How the live system is doing: health numbers, alerts, and the answers
    worth reviewing (low judge score or thumbs down)."""
    days = min(max(days, 1), 90)
    report = online_report.summary(conn, days)
    return {"summary": report, "alerts": online_report.alerts(report), "review_queue": online_report.review_queue(conn, days)}


@app.get("/stats")
def stats(admin: dict = Depends(admin_user)):
    """Request count, latency percentiles over the last 1000 chat requests,
    and cumulative Gemini token usage/cost since the process started."""
    ordered = sorted(_latencies)
    return {
        "requests": len(ordered),
        "latency_s": {
            "avg": round(sum(ordered) / len(ordered), 3) if ordered else 0.0,
            "p50": percentile(ordered, 0.50),
            "p95": percentile(ordered, 0.95),
            "max": ordered[-1] if ordered else 0.0,
        },
        "usage": usage_tracker.snapshot(),
        "llm_providers": llm_connection.status(),
    }
