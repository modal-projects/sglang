from types import SimpleNamespace

from sglang.srt.models import registry


def test_model_discovery_does_not_import_sibling_tests(monkeypatch):
    package_name = "registry_test_models"
    imports = []

    class SupportedModel:
        pass

    def import_module(name):
        imports.append(name)
        if name == package_name:
            return SimpleNamespace(__path__=["models"])
        if name == f"{package_name}.model":
            return SimpleNamespace(EntryClass=SupportedModel)
        raise AssertionError(f"Unexpected model import: {name}")

    monkeypatch.setattr(registry.importlib, "import_module", import_module)
    monkeypatch.setattr(
        registry.pkgutil,
        "iter_modules",
        lambda *_: [
            (None, f"{package_name}.model", False),
            (None, f"{package_name}.model_test", False),
            (None, f"{package_name}.subpackage", True),
        ],
    )
    registry.import_model_classes.cache_clear()

    try:
        models = registry.import_model_classes(package_name, strict=True)
    finally:
        registry.import_model_classes.cache_clear()

    assert models == {"SupportedModel": SupportedModel}
    assert imports == [package_name, f"{package_name}.model"]
