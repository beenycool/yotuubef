import sys
import unittest.mock

sys.modules['psutil'] = unittest.mock.MagicMock()
sys.modules['aiohttp'] = unittest.mock.MagicMock()
sys.modules['yaml'] = unittest.mock.MagicMock()
sys.modules['numpy'] = unittest.mock.MagicMock()
sys.modules['moviepy'] = unittest.mock.MagicMock()
sys.modules['asyncprawcore'] = unittest.mock.MagicMock()
sys.modules['asyncprawcore.exceptions'] = unittest.mock.MagicMock()
sys.modules['dotenv'] = unittest.mock.MagicMock()

# Do not mock pydantic because typing fails when Field returns a mock object that can't be used in List[VideoSegment].
# We need real pydantic. Let's see if we can install it temporarily for testing in the script if it's not present.

import pytest
pytest.main(['tests/test_hybrid_script_guards.py'])
