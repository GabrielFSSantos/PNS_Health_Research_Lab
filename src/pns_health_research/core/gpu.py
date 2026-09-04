from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import subprocess

logger = logging.getLogger("pns_health_research")

_CACHED: bool | None = None
_XGB_GPU_CACHED: bool | None = None


def cuda_available(*, refresh: bool = False) -> bool:
    """Detect CUDA runtime without relying solely on nvidia-smi (blocked in some sandboxes)."""
    global _CACHED
    if not refresh and _CACHED is not None:
        return _CACHED

    if os.environ.get("PNS_DISABLE_CUDA") == "1":
        _CACHED = False
        return False
    if os.environ.get("PNS_FORCE_CUDA") == "1":
        _CACHED = True
        return True

    if _libcudart_loadable():
        _CACHED = True
        return True

    try:
        import torch

        if torch.cuda.is_available():
            _CACHED = True
            return True
    except ImportError:
        pass

    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            _CACHED = True
            return True
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    _CACHED = False
    return False


def _libcudart_loadable() -> bool:
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    for name in candidates:
        if not name:
            continue
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def gpu_ready_for_training(*, refresh: bool = False) -> bool:
    """CUDA utilizável pelo XGBoost (evita device=cuda sem GPU visível)."""
    global _XGB_GPU_CACHED
    if not refresh and _XGB_GPU_CACHED is not None:
        return _XGB_GPU_CACHED
    if not cuda_available(refresh=refresh):
        _XGB_GPU_CACHED = False
        return False
    try:
        import numpy as np
        import xgboost as xgb

        x = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.float32)
        clf = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        clf.fit(x, y)
        _XGB_GPU_CACHED = True
    except Exception:
        _XGB_GPU_CACHED = False
    return _XGB_GPU_CACHED


def cuda_device_label() -> str:
    if not cuda_available():
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "cuda"
