"""Server entry point for Proxima Auto-Scaler.

Provides the ``serve`` console script and a ``main()`` function
that launches the FastAPI application via uvicorn.
"""

from __future__ import annotations

import uvicorn

from app import app


def main() -> None:
    """Launch the Proxima Auto-Scaler API server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
