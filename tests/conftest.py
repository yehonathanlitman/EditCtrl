# tests/conftest.py
import sys
import types

# Stub out optional heavy dependencies that are not needed for unit tests
# but are imported at the top of diffsynth/__init__.py's import chain.
if "modelscope" not in sys.modules:
    _ms = types.ModuleType("modelscope")
    _ms.snapshot_download = lambda *a, **kw: None  # only attribute used at import time
    sys.modules["modelscope"] = _ms

import torch
import pytest

@pytest.fixture
def cpu_device():
    return torch.device("cpu")

@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(0)
