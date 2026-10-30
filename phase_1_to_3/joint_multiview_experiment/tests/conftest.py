import os
import shutil
import sys
import tempfile
from pathlib import Path
import pytest

# Add experiment root to sys.path
EXP_ROOT = str(Path(__file__).resolve().parent.parent)
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)

# Ensure EXP_STATE_DIR is set immediately on import
if "EXP_STATE_DIR" not in os.environ:
    os.environ["EXP_STATE_DIR"] = tempfile.mkdtemp(prefix="test_isolated_state_global_")


@pytest.fixture(autouse=True, scope="session")
def isolate_test_state_dir():
    """
    Guarantees that test executions NEVER write to the real state/ directory.
    Sets EXP_STATE_DIR to an isolated temporary directory for the duration of tests.
    """
    tmp_state = tempfile.mkdtemp(prefix="test_isolated_state_")
    old_state = os.environ.get("EXP_STATE_DIR")
    os.environ["EXP_STATE_DIR"] = tmp_state
    yield tmp_state
    shutil.rmtree(tmp_state, ignore_errors=True)
    if old_state:
        os.environ["EXP_STATE_DIR"] = old_state
