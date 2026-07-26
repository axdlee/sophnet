from unittest.mock import patch

import pytest

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model.message import (
    TextPromptMessageContent,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from dify_plugin.errors.model import CredentialsValidateFailedError

from models.llm.llm import SophnetLargeLanguageModel
from models.text_embedding.text_embedding import SophnetTextEmbeddingModel
from models.tts.tts import SophnetText2SpeechModel
from provider.sophnet import SophNetProvider


def test_llm_openai_converter_preserves_video_content() -> None:
    model = object.__new__(SophnetLargeLanguageModel)
    message = UserPromptMessage(
        content=[
            TextPromptMessageContent(data="Describe this video"),
            VideoPromptMessageContent(
                format="mp4",
                mime_type="video/mp4",
                url="https://example.com/demo.mp4",
            ),
        ]
    )

    assert model._convert_prompt_message_to_dict(message) == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video"},
            {
                "type": "video_url",
                "video_url": {"url": "https://example.com/demo.mp4"},
            },
        ],
    }


def test_llm_token_count_delegates_to_sdk() -> None:
    model = object.__new__(SophnetLargeLanguageModel)

    with patch.object(OAICompatLargeLanguageModel, "get_num_tokens", return_value=37) as counter:
        result = model.get_num_tokens("DeepSeek-V4-Flash", {}, [])

    assert result == 37
    counter.assert_called_once_with("DeepSeek-V4-Flash", {}, [], None)


@pytest.mark.parametrize(
    ("legacy_model", "replacement_model"),
    [
        ("DeepSeek-V3.1", "DeepSeek-V4-Flash"),
        ("DeepSeek-R1", "DeepSeek-V4-Pro"),
        ("qwen-vl-plus-latest", "qwen3-vl-plus"),
    ],
)
def test_llm_legacy_aliases_apply_to_token_count_and_validation(
    legacy_model: str, replacement_model: str
) -> None:
    model = object.__new__(SophnetLargeLanguageModel)
    credentials = {"api_key": "key"}

    with (
        patch.object(OAICompatLargeLanguageModel, "get_num_tokens", return_value=1) as counter,
        patch.object(OAICompatLargeLanguageModel, "validate_credentials") as validator,
    ):
        model.get_num_tokens(legacy_model, credentials, [])
        model.validate_credentials(legacy_model, credentials)

    counter.assert_called_once_with(replacement_model, credentials, [], None)
    assert validator.call_args.args[0] == replacement_model
    assert credentials == {"api_key": "key"}


def test_provider_validation_uses_current_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    validator_calls: list[tuple[str, dict]] = []

    class FakeModel:
        def validate_credentials(self, model: str, credentials: dict) -> None:
            validator_calls.append((model, credentials))

    provider = object.__new__(SophNetProvider)
    monkeypatch.setattr(SophNetProvider, "get_model_instance", lambda self, model_type: FakeModel())
    credentials = {"api_key": "test", "project_id": "project"}

    provider.validate_provider_credentials(credentials)

    assert validator_calls == [("DeepSeek-V4-Flash", credentials)]


def test_embedding_dimensions_are_model_specific() -> None:
    assert SophnetTextEmbeddingModel._validate_dimensions("text-embeddings", 768) == 768
    assert SophnetTextEmbeddingModel._validate_dimensions("clip-embeddings", 64) == 64
    assert SophnetTextEmbeddingModel._validate_dimensions("bge-m3", 1024) == 1024

    with pytest.raises(ValueError, match="bge-m3 only supports 1024"):
        SophnetTextEmbeddingModel._validate_dimensions("bge-m3", 768)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", True), ("false", False), ("1", True), ("0", False)],
)
def test_tts_streaming_boolean_normalization(value: object, expected: bool) -> None:
    assert SophnetText2SpeechModel._as_bool(value) is expected


def test_tts_streaming_boolean_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="streaming must be a boolean"):
        SophnetText2SpeechModel._as_bool("sometimes")


def test_provider_keeps_standard_credential_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def validate_credentials(self, model: str, credentials: dict) -> None:
            raise CredentialsValidateFailedError("invalid api key")

    provider = object.__new__(SophNetProvider)
    monkeypatch.setattr(SophNetProvider, "get_model_instance", lambda self, model_type: FakeModel())

    with pytest.raises(CredentialsValidateFailedError, match="invalid api key"):
        provider.validate_provider_credentials({})
