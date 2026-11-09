import pytest
from pydantic import SecretStr, ValidationError

from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.llm.openai_provider import OpenAIProvider

TEST_SECRET = "test-signing-key-that-is-long-enough-32"


def make_settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        auth_secret_key=SecretStr(TEST_SECRET),
        **overrides,
    )


def test_production_rejects_the_public_default_secret():
    with pytest.raises(ValidationError, match="public default"):
        Settings(_env_file=None, app_env="production")


def test_production_rejects_the_env_example_placeholder():
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(
            _env_file=None,
            app_env="production",
            auth_secret_key=SecretStr("replace-with-a-random-secret-at-least-32-characters"),
        )


def test_production_rejects_a_short_secret():
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(
            _env_file=None,
            app_env="production",
            auth_secret_key=SecretStr("too-short"),
        )


def test_development_allows_a_short_secret():
    settings = Settings(
        _env_file=None,
        app_env="development",
        auth_secret_key=SecretStr("short"),
    )

    assert settings.auth_secret_key.get_secret_value() == "short"


def test_default_environment_is_production():
    assert Settings.model_fields["app_env"].default == "production"


def test_cookie_secure_is_automatic_outside_localhost():
    settings = make_settings()

    assert settings.cookie_secure("hr.example.org") is True
    assert settings.cookie_secure(None) is True
    assert settings.cookie_secure("localhost") is False
    assert settings.cookie_secure("127.0.0.1") is False
    assert settings.cookie_secure("::1") is False


def test_explicit_cookie_secure_wins_over_host_detection():
    settings = make_settings(auth_cookie_secure=False)

    assert settings.cookie_secure("hr.example.org") is False


def test_integration_flags_follow_credentials():
    settings = make_settings(
        openai_api_key=SecretStr("openai-key"),
        langfuse_public_key=SecretStr("public"),
        langfuse_secret_key=SecretStr("secret"),
        smtp_host="smtp.example.org",
        smtp_from="hr@example.org",
    )

    assert settings.openai_configured is True
    assert settings.langfuse_configured is True
    assert settings.smtp_configured is True


def test_openai_provider_is_disabled_without_a_key():
    provider = OpenAIProvider(make_settings())

    assert provider.configured is False
    assert provider.client is None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        provider.require_client()


async def test_embed_texts_is_a_noop_for_an_empty_batch():
    provider = OpenAIProvider(make_settings())

    assert await provider.embed_texts([]) == []


def test_reasoning_switch_is_sent_unless_explicitly_enabled():
    disabled = OpenAIProvider(make_settings())
    enabled = OpenAIProvider(make_settings(chat_reasoning_enabled=True))

    assert disabled.chat_extra_body == {"reasoning": {"enabled": False}}
    assert enabled.chat_extra_body == {}
