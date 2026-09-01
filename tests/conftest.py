import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "regression: testes que exigem PNS real")
