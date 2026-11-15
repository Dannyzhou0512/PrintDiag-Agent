import importlib

import pytest


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in ("MARLIN_BASE_URL", "MARLIN_HTTP_TIMEOUT", "LOG_LEVEL"):
        monkeypatch.delenv(key, raising=False)


def test_package_import():
    assert importlib.import_module("printdiag").__name__ == "printdiag"


def test_config_defaults():
    from printdiag.config import load_config

    config = load_config()

    assert config.marlin_base_url == "http://127.0.0.1:8100"
    assert config.marlin_http_timeout == 5.0
    assert config.log_level == "INFO"


def test_dotenv_values(tmp_path):
    from printdiag.config import load_config

    (tmp_path / ".env").write_text(
        "MARLIN_BASE_URL=http://127.0.0.1:8101\n"
        "MARLIN_HTTP_TIMEOUT=2.5\n"
        "LOG_LEVEL=WARNING\n",
        encoding="utf-8",
    )

    config = load_config()

    assert config.marlin_base_url == "http://127.0.0.1:8101"
    assert config.marlin_http_timeout == 2.5
    assert config.log_level == "WARNING"


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    from printdiag.config import load_config

    (tmp_path / ".env").write_text(
        "MARLIN_BASE_URL=http://127.0.0.1:8101\n"
        "MARLIN_HTTP_TIMEOUT=5\n"
        "LOG_LEVEL=INFO\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MARLIN_BASE_URL", "http://127.0.0.1:8102")
    monkeypatch.setenv("MARLIN_HTTP_TIMEOUT", "1.5")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    config = load_config()

    assert config.marlin_base_url == "http://127.0.0.1:8102"
    assert config.marlin_http_timeout == 1.5
    assert config.log_level == "DEBUG"


@pytest.mark.parametrize("value", ["", "abc", "0", "-1", "nan", "inf", "-inf"])
def test_invalid_timeout(monkeypatch, value):
    from printdiag.config import load_config

    monkeypatch.setenv("MARLIN_HTTP_TIMEOUT", value)

    with pytest.raises(
        ValueError, match="MARLIN_HTTP_TIMEOUT must be a finite number greater than 0"
    ):
        load_config()
