"""
app.py — FastAPI server exposing the DataQuality environment as a REST API.
Deployed on Hugging Face Spaces. Implements OpenEnv HTTP spec.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from dataquality_env import Action, DataQualityEnv

app = FastAPI(
    title="DataQuality OpenEnv",
    description="Real-world data quality triage environment for AI agents.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (for demo; production would use Redis)
_sessions: Dict[str, DataQualityEnv] = {}


class CreateSessionRequest(BaseModel):
    task_id: str = "task_easy"
    seed: int = 42


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html><body style="font-family:monospace;padding:2rem;background:#0d1117;color:#c9d1d9">
    <h1>🧹 DataQuality OpenEnv</h1>
    <p>Real-world data quality triage environment for AI agents.</p>
    <h2>Quick Start</h2>
    <pre>
# 1. Create a session
POST /env/reset  {"task_id": "task_easy", "seed": 42}

# 2. Step
POST /env/step   {"session_id": "...", "action": {"action_type": "inspect_column", "parameters": {}}}

# 3. Get state
GET  /env/state/{session_id}
    </pre>
    <p><a href="/docs" style="color:#58a6ff">→ Interactive API Docs</a></p>
    <p><a href="/env/info" style="color:#58a6ff">→ Environment Info</a></p>
    </body></html>
    """


@app.get("/env/info")
def env_info():
    return {
        "name": "data-quality-triage",
        "version": "1.0.0",
        "tasks": [
            {"id": "task_easy", "difficulty": "easy", "max_steps": 15, "description": "Fix missing values and duplicates in a contacts CSV"},
            {"id": "task_medium", "difficulty": "medium", "max_steps": 20, "description": "Fix type errors, date formats, negatives, duplicate transaction IDs"},
            {"id": "task_hard", "difficulty": "hard", "max_steps": 30, "description": "Multi-issue healthcare dataset: missing IDs, format violations, outliers, bad categoricals"},
        ],
        "action_types": [
            "inspect_column", "fill_missing", "drop_duplicates", "fix_type",
            "flag_outlier", "apply_regex_fix", "drop_rows", "rename_column", "submit"
        ],
        "openenv_spec": "1.0",
    }


@app.post("/env/reset")
def reset(req: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    env = DataQualityEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions[session_id] = env
    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
    }


@app.post("/env/step")
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found. Call /env/reset first.")

    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/env/state/{session_id}")
def state(session_id: str):
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state().model_dump()


@app.delete("/env/session/{session_id}")
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return {"deleted": session_id}


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}
