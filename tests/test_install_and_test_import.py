import importlib.util
from pathlib import Path


def test_install_and_test_module_import_is_side_effect_free():
    module_path = Path(__file__).resolve().parent.parent / "install_and_test.py"
    spec = importlib.util.spec_from_file_location(
        "install_and_test_module", module_path
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert getattr(module, "__test__", False) is False
    assert callable(module.main)
