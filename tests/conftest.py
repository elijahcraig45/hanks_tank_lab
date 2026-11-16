import pytest


@pytest.fixture(autouse=True)
def _blocked_account_domain(monkeypatch):
    """Every test runs with one made-up blocked domain, so the work-account guards are exercised
    without any real domain appearing in the code."""
    monkeypatch.setenv("MLLAB_BLOCKED_ACCOUNT_DOMAINS", "blocked.example")
