import sys
import unittest.mock

__test__ = False


def main() -> int:
    sys.modules["psutil"] = unittest.mock.MagicMock()
    sys.modules["aiohttp"] = unittest.mock.MagicMock()
    sys.modules["yaml"] = unittest.mock.MagicMock()
    sys.modules["numpy"] = unittest.mock.MagicMock()
    sys.modules["moviepy"] = unittest.mock.MagicMock()
    sys.modules["asyncprawcore"] = unittest.mock.MagicMock()
    sys.modules["asyncprawcore.exceptions"] = unittest.mock.MagicMock()
    sys.modules["dotenv"] = unittest.mock.MagicMock()

    import pytest

    return pytest.main(["tests/test_hybrid_script_guards.py"])


if __name__ == "__main__":
    sys.exit(main())
