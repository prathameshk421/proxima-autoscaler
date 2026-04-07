"""Compatibility helpers for Stable-Baselines3 checkpoint loading on Windows/PyTorch."""

from __future__ import annotations

import io
import zipfile
from typing import Any

import torch


_PATCHED = False
_ORIGINAL_TORCH_LOAD = torch.load


def patch_torch_load_for_zip_stream() -> None:
    """Patch torch.load to safely read non-seekable zip streams via in-memory buffer.

    SB3 loads tensors from ZipExtFile handles. Some torch versions on Windows may
    fail with `PytorchStreamReader failed reading file .data/serialization_id` when
    using non-seekable zip streams. This patch converts those streams to BytesIO.
    """
    global _PATCHED
    if _PATCHED:
        return

    def _patched_torch_load(f: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(f, zipfile.ZipExtFile):
            return _ORIGINAL_TORCH_LOAD(io.BytesIO(f.read()), *args, **kwargs)

        if hasattr(f, "read") and not hasattr(f, "seek"):
            return _ORIGINAL_TORCH_LOAD(io.BytesIO(f.read()), *args, **kwargs)

        return _ORIGINAL_TORCH_LOAD(f, *args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]
    _PATCHED = True