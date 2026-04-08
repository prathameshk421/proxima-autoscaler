"""server/app.py — OpenEnv server entry point.

Re-exports the root FastAPI application and provides the main()
entry point required by openenv validate.
"""

import uvicorn

from app import app  # noqa: F401  (re-export for openenv)


def main():
    """Start the Proxima Auto-Scaler API server."""
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
