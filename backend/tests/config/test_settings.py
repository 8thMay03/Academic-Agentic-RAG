from app.config.settings import Settings


def test_settings_loads_api_keys_from_secret_files(tmp_path) -> None:
    openai_secret = tmp_path / "openai_api_key"
    tavily_secret = tmp_path / "tavily_api_key"
    api_secret = tmp_path / "api_key"
    openai_secret.write_text("openai-from-file\n", encoding="utf-8")
    tavily_secret.write_text("tavily-from-file\n", encoding="utf-8")
    api_secret.write_text("api-from-file\n", encoding="utf-8")

    settings = Settings(
        OPENAI_API_KEY_FILE=str(openai_secret),
        TAVILY_API_KEY_FILE=str(tavily_secret),
        API_KEY_FILE=str(api_secret),
        _env_file=None,
    )

    assert settings.OPENAI_API_KEY == "openai-from-file"
    assert settings.TAVILY_API_KEY == "tavily-from-file"
    assert settings.API_KEY == "api-from-file"


def test_settings_direct_secret_value_wins_over_secret_file(tmp_path) -> None:
    secret_file = tmp_path / "openai_api_key"
    secret_file.write_text("from-file", encoding="utf-8")

    settings = Settings(
        OPENAI_API_KEY="from-env",
        OPENAI_API_KEY_FILE=str(secret_file),
        _env_file=None,
    )

    assert settings.OPENAI_API_KEY == "from-env"
