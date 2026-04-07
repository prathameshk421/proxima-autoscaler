"""FastAPI app for Hugging Face Spaces deployment of PPO autoscaling model."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from stable_baselines3 import PPO

from src.agent.sb3_compat import patch_torch_load_for_zip_stream
from src.evaluation.evaluator import evaluate_baselines_for_seed, evaluate_ppo_run
from src.env.grader import grade_episode

app = FastAPI(title="Cloud AutoScaling RL", version="0.1.0")

MODEL_LOAD_ERROR: str | None = None
try:
    patch_torch_load_for_zip_stream()
    model = PPO.load("checkpoints/ppo_cloud_final", device="cpu")
except Exception as exc:  # pragma: no cover - startup dependency/runtime issue
    model = None
    MODEL_LOAD_ERROR = str(exc)


@app.get("/health")
def health() -> dict[str, Any]:
    """Health endpoint indicating whether PPO model loaded successfully."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/evaluate")
def evaluate(
    difficulty: str = Query(default="easy", pattern="^(easy|medium|hard)$"),
    seed: int = Query(default=42, ge=0),
) -> dict[str, Any]:
    """Evaluate one PPO episode for selected difficulty and seed."""
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")

    try:
        metrics = evaluate_ppo_run(difficulty=difficulty, seed=seed, model=model)
        score = grade_episode(
            {
                "avg_latency": 0.0,
                "avg_reward": float(metrics["avg_reward"]),
                "oscillation_count": float(metrics["oscillation_count"]),
                "steps": float(metrics["steps"]),
                "signal_tp": float(metrics["capacity_signal_accuracy"] * 10.0),
                "signal_fp": float((1.0 - metrics["capacity_signal_accuracy"]) * 5.0),
                "signal_fn": 0.0,
                "avg_servers": 1.0 + 9.0 * (1.0 - float(metrics["capacity_waste"])),
                "min_servers": 1.0,
                "max_servers": 10.0,
                "sla_threshold": 220.0,
            },
            difficulty=difficulty,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return {
        "difficulty": difficulty,
        "seed": seed,
        "score": float(score),
        "sla_violation_rate": float(metrics["sla_violation_rate"]),
        "avg_reward": float(metrics["avg_reward"]),
        "steps": int(metrics["steps"]),
    }


@app.get("/baselines")
def baselines(
    difficulty: str = Query(default="easy", pattern="^(easy|medium|hard)$"),
    seed: int = Query(default=42, ge=0),
) -> dict[str, Any]:
    """Run all heuristics baselines for selected difficulty and seed."""
    try:
        baseline_metrics = evaluate_baselines_for_seed(difficulty=difficulty, seed=seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Baseline evaluation failed: {exc}") from exc

    serializable = {
        name: {k: float(v) for k, v in values.items()}
        for name, values in baseline_metrics.items()
    }

    return {
        "difficulty": difficulty,
        "seed": seed,
        "baselines": serializable,
    }
