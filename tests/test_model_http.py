from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.text_embedding import EmbeddingUsage
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeBadRequestError

from models.llm.llm import SophnetLargeLanguageModel
from models.speech2text.speech2text import SophnetSpeech2TextModel
from models.text_embedding.text_embedding import SophnetTextEmbeddingModel
from models.tts.tts import SophnetText2SpeechModel


class FakeResponse:
    def __init__(
        self,
        data: dict | None = None,
        *,
        content: bytes = b"",
        lines: list[str] | None = None,
    ) -> None:
        self._data = data or {}
        self.content = content
        self._lines = lines or []
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data

    def iter_lines(self, decode_unicode: bool = False) -> Iterator[str]:
        assert decode_unicode is True
        yield from self._lines

    def close(self) -> None:
        self.closed = True


def embedding_usage(tokens: int) -> EmbeddingUsage:
    return EmbeddingUsage(
        tokens=tokens,
        total_tokens=tokens,
        unit_price=Decimal(0),
        price_unit=Decimal(0),
        total_price=Decimal(0),
        currency="RMB",
        latency=0,
    )


def test_llm_invoke_uses_enriched_copy_without_mutating_credentials() -> None:
    model = object.__new__(SophnetLargeLanguageModel)
    credentials = {"api_key": "key"}
    schema = SimpleNamespace(features=[ModelFeature.TOOL_CALL])

    with (
        patch.object(SophnetLargeLanguageModel, "get_model_schema", return_value=schema),
        patch.object(OAICompatLargeLanguageModel, "_invoke", return_value="ok") as invoke,
    ):
        result = model._invoke("DeepSeek-V4-Flash", credentials, [], {})

    assert result == "ok"
    assert credentials == {"api_key": "key"}
    request_credentials = invoke.call_args.args[1]
    assert request_credentials["endpoint_url"] == SophnetLargeLanguageModel.endpoint_url
    assert request_credentials["function_calling_type"] == "tool_call"
    assert request_credentials["stream_function_calling"] == "supported"


@pytest.mark.parametrize(
    ("legacy_model", "replacement_model"),
    [
        ("DeepSeek-V3.1", "DeepSeek-V4-Flash"),
        ("DeepSeek-R1", "DeepSeek-V4-Pro"),
        ("qwen-vl-plus-latest", "qwen3-vl-plus"),
    ],
)
def test_llm_invoke_maps_retired_models_to_supported_replacements(
    legacy_model: str, replacement_model: str
) -> None:
    model = object.__new__(SophnetLargeLanguageModel)

    with (
        patch.object(
            SophnetLargeLanguageModel,
            "get_model_schema",
            return_value=SimpleNamespace(features=[]),
        ),
        patch.object(OAICompatLargeLanguageModel, "_invoke", return_value="ok") as invoke,
    ):
        result = model._invoke(legacy_model, {"api_key": "key"}, [], {})

    assert result == "ok"
    assert invoke.call_args.args[0] == replacement_model


def test_embedding_batches_and_preserves_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(SophnetTextEmbeddingModel)
    vectors = [[float(index)] * 64 for index in range(1, 4)]
    responses = iter(
        [
            FakeResponse(
                {
                    "data": [{"embedding": vectors[0]}, {"embedding": vectors[1]}],
                    "usage": {"total_tokens": 4},
                }
            ),
            FakeResponse({"data": [{"embedding": vectors[2]}], "usage": {"total_tokens": 2}}),
        ]
    )
    calls: list[dict] = []

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        calls.append(kwargs["json"])
        return next(responses)

    monkeypatch.setattr("models.text_embedding.text_embedding.requests.post", fake_post)
    monkeypatch.setattr(SophnetTextEmbeddingModel, "_get_context_size", lambda *args: 8192)
    monkeypatch.setattr(SophnetTextEmbeddingModel, "_get_max_chunks", lambda *args: 2)
    monkeypatch.setattr(SophnetTextEmbeddingModel, "_get_num_tokens_by_gpt2", lambda *args: 1)
    monkeypatch.setattr(
        SophnetTextEmbeddingModel,
        "_calc_response_usage",
        lambda self, model, credentials, tokens: embedding_usage(tokens),
    )

    result = model._invoke(
        "text-embeddings",
        {"api_key": "key", "project_id": "project", "dimensions": 64},
        ["one", "two", "three"],
    )

    assert result.embeddings == vectors
    assert result.usage.total_tokens == 6
    assert [call["input_texts"] for call in calls] == [["one", "two"], ["three"]]


def test_embedding_rejects_incomplete_response(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse({"data": []})
    monkeypatch.setattr(
        "models.text_embedding.text_embedding.requests.post",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(InvokeBadRequestError, match="0 vectors for 1 inputs"):
        data = SophnetTextEmbeddingModel._request_embeddings(
            "https://example.invalid", {}, {"input_texts": ["one"]}
        )
        SophnetTextEmbeddingModel._extract_embeddings(data, 1)
    assert response.closed is True


def test_stt_polling_waits_then_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(SophnetSpeech2TextModel)
    first_response = FakeResponse({"status": "waiting"})
    second_response = FakeResponse({"status": "success", "result": "done"})
    responses = iter([first_response, second_response])
    sleeps: list[int] = []
    monkeypatch.setattr(
        "models.speech2text.speech2text.requests.get", lambda *args, **kwargs: next(responses)
    )
    monkeypatch.setattr("models.speech2text.speech2text.time.sleep", sleeps.append)

    result = model._poll_task_result(
        {"project_id": "project", "api_key": "key"}, "task", max_retries=2, interval=1
    )

    assert result == "done"
    assert sleeps == [1]
    assert first_response.closed is True
    assert second_response.closed is True


def test_stt_validation_file_is_temporary_and_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(SophnetSpeech2TextModel)
    created_path: Path | None = None
    original_create = SophnetSpeech2TextModel._create_validation_audio
    response = FakeResponse({"task_id": "task"})

    def track_path() -> str:
        nonlocal created_path
        created_path = Path(original_create())
        assert created_path.exists()
        assert "models/speech2text" not in created_path.as_posix()
        return str(created_path)

    monkeypatch.setattr(
        SophnetSpeech2TextModel, "_create_validation_audio", staticmethod(track_path)
    )
    monkeypatch.setattr(
        "models.speech2text.speech2text.requests.post",
        lambda *args, **kwargs: response,
    )

    model.validate_credentials(
        "Paraformer",
        {"project_id": "project", "api_key": "key", "easyllm_id": "deployment"},
    )

    assert created_path is not None
    assert not created_path.exists()
    assert response.closed is True


def test_stt_missing_credentials_do_not_create_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(SophnetSpeech2TextModel)
    monkeypatch.setattr(
        SophnetSpeech2TextModel,
        "_create_validation_audio",
        staticmethod(lambda: pytest.fail("validation audio must not be created")),
    )

    with pytest.raises(CredentialsValidateFailedError, match="project_id is required"):
        model.validate_credentials("Paraformer", {})


def test_tts_stream_decodes_sse_and_closes_response(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(SophnetText2SpeechModel)
    response = FakeResponse(
        lines=["event: message", 'data: {"audioFrame": "aGk="}', "data: [DONE]"]
    )
    monkeypatch.setattr("models.tts.tts.requests.post", lambda *args, **kwargs: response)

    result = list(
        model._invoke_streaming(
            "cosyvoice-v2",
            {"project_id": "project", "api_key": "key", "easyllm_id": "deployment"},
            ["hello"],
            "longxiaochun",
        )
    )

    assert result == [b"hi"]
    assert response.closed is True


def test_tts_false_string_uses_non_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(SophnetText2SpeechModel)
    monkeypatch.setattr(
        SophnetText2SpeechModel,
        "_invoke_streaming",
        lambda *args, **kwargs: pytest.fail("streaming path must not run"),
    )
    monkeypatch.setattr(
        SophnetText2SpeechModel,
        "_invoke_non_streaming",
        lambda *args, **kwargs: iter([b"audio"]),
    )

    result = list(
        model._invoke(
            "cosyvoice-v2",
            "tenant",
            {"streaming": "false"},
            "hello",
            "longxiaochun",
        )
    )

    assert result == [b"audio"]
