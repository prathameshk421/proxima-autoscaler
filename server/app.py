from __future__ import annotations

import uvicorn

from server.main import app


def main() -> None:
    """Start the OpenEnv server entrypoint."""
    uvicorn.run("server.main:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
