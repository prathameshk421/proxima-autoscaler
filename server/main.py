"""
Phase 5 — FastAPI Server

HTTP wrapper for the Cloud Auto-Scaling OpenEnv with UUID-based session
management.  Each ``/reset`` call creates a fresh environment keyed by a
unique session ID, preventing concurrent graders from clobbering each other.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query

from env.environment import CloudAutoScalingEnv
from env.models import CloudAction, CloudObservation, StepResult

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CloudAutoScaling OpenEnv",
    description=(
        "An OpenEnv-compliant RL environment that simulates intelligent "
        "cloud infrastructure auto-scaling."
    ),
    version="1.0.0",
)

# Session store — keyed by UUID, NOT task name.
# This prevents concurrent grader threads from clobbering each other.
sessions: dict[str, CloudAutoScalingEnv] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> CloudAutoScalingEnv:
    """Look up a session or raise 404."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    """Welcome endpoint for browser/root URL visits."""
    return {
        "message": "Welcome to proxima-autoscalar",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "reset": "POST /reset?task=easy|medium|hard",
            "step": "POST /step?session_id=<uuid>",
            "state": "GET /state?session_id=<uuid>",
        },
    }

@app.post("/reset")
def reset(task: str = Query("easy", description="Task name: easy, medium, or hard")) -> dict:
    """Start a new episode.  Returns a ``session_id`` for subsequent calls."""
    try:
        env = CloudAutoScalingEnv(task_name=task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    session_id = str(uuid4())
    sessions[session_id] = env
    obs = env.reset()
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step")
def step(
    action: CloudAction,
    session_id: str = Query(..., description="Session UUID from /reset"),
) -> StepResult:
    """Take one action and receive the next observation, reward, and done flag."""
    env = _get_session(session_id)
    try:
        result = env.step(action)
    except RuntimeError as exc:
        # Episode already done
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/state")
def state(
    session_id: str = Query(..., description="Session UUID from /reset"),
) -> CloudObservation:
    """Get the current observation without advancing the step."""
    env = _get_session(session_id)
    return env.state()


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
