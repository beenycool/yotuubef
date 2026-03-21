import sys
import subprocess

try:
    import openai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])

import pytest
sys.exit(pytest.main(['tests/test_hybrid_script_guards.py']))
