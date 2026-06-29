import subprocess
import sys

__test__ = False


def main() -> int:
    try:
        import openai  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])

    import pytest

    return pytest.main(["tests/test_hybrid_script_guards.py"])


if __name__ == "__main__":
    sys.exit(main())
