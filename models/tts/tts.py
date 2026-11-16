import base64
import json
from collections.abc import Generator, Mapping
from typing import Any

import requests
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
    PriceConfig,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
)
from dify_plugin.interfaces.model.tts_model import TTSModel


class SophnetText2SpeechModel(TTSModel):
    """SophNet text-to-speech model with streaming and non-streaming APIs."""

    DEFAULT_FORMAT = "MP3_16000HZ_MONO_128KBPS"
    DEFAULT_MODEL = "cosyvoice-v2"
    DEFAULT_VOICE = "longxiaochun"
    SUPPORTED_VOICES = [
        {"name": "龙小春", "mode": "longxiaochun", "language": "zh-Hans"},
        {"name": "龙小夏", "mode": "longxiaoxia", "language": "zh-Hans"},
        {"name": "龙小成", "mode": "longxiaocheng", "language": "zh-Hans"},
        {"name": "龙小白", "mode": "longxiaobai", "language": "zh-Hans"},
        {"name": "龙老铁", "mode": "longlaotie", "language": "zh-Hans"},
        {"name": "龙叔", "mode": "longshu", "language": "zh-Hans"},
        {"name": "龙硕", "mode": "longshuo", "language": "zh-Hans"},
        {"name": "龙婧", "mode": "longjing", "language": "zh-Hans"},
        {"name": "龙悦", "mode": "longyue", "language": "zh-Hans"},
        {"name": "龙湾", "mode": "longwan", "language": "zh-Hans"},
        {"name": "龙成", "mode": "longcheng", "language": "zh-Hans"},
        {"name": "龙华", "mode": "longhua", "language": "zh-Hans"},
        {"name": "Stella", "mode": "loongstella", "language": "en-US"},
        {"name": "Bella", "mode": "loongbella", "language": "en-US"},
    ]
    SUPPORTED_FORMATS = [
        "MP3_16000HZ_MONO_128KBPS",
        "MP3_24000HZ_MONO_128KBPS",
        "MP3_48000HZ_MONO_128KBPS",
        "WAV_16000HZ_MONO_16BIT",
        "WAV_24000HZ_MONO_16BIT",
        "WAV_48000HZ_MONO_16BIT",
    ]

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: str | None = None,
    ) -> Generator[bytes, None, None]:
        del tenant_id, user
        valid_voices = {item["mode"] for item in self.SUPPORTED_VOICES}
        selected_voice = voice if voice in valid_voices else self.DEFAULT_VOICE
        streaming = self._as_bool(credentials.get("streaming", True))
        texts = (
            self._split_text_into_sentences(content_text, max_length=500)
            if len(content_text) > 500
            else [content_text]
        )
        if streaming:
            yield from self._invoke_streaming(model, credentials, texts, selected_voice)
        else:
            yield from self._invoke_non_streaming(model, credentials, texts, selected_voice)

    def _invoke_streaming(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        voice: str,
    ) -> Generator[bytes, None, None]:
        url, headers, payload = self._build_request_params(
            credentials, model, texts, voice, streaming=True
        )
        response: requests.Response | None = None
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                prefix, separator, raw_data = line.partition(":")
                if separator != ":" or prefix.strip() != "data":
                    continue
                raw_data = raw_data.strip()
                if not raw_data or raw_data == "[DONE]":
                    continue
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue
                audio_frame = data.get("audioFrame") if isinstance(data, dict) else None
                if audio_frame:
                    yield base64.b64decode(audio_frame, validate=True)
        except requests.RequestException as ex:
            raise InvokeConnectionError("SophNet TTS streaming request failed") from ex
        except (ValueError, TypeError) as ex:
            raise InvokeBadRequestError("SophNet TTS stream returned invalid audio data") from ex
        finally:
            if response is not None:
                response.close()

    def _invoke_non_streaming(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        voice: str,
    ) -> Generator[bytes, None, None]:
        url, headers, payload = self._build_request_params(
            credentials, model, texts, voice, streaming=False
        )
        response: requests.Response | None = None
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            yield response.content
        except requests.RequestException as ex:
            raise InvokeConnectionError("SophNet TTS request failed") from ex
        finally:
            if response is not None:
                response.close()

    def validate_credentials(self, model: str, credentials: Mapping) -> None:
        try:
            url, headers, payload = self._build_request_params(
                dict(credentials), model, ["你好，这是一个测试"], self.DEFAULT_VOICE, False
            )
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            try:
                response.raise_for_status()
            finally:
                response.close()
        except CredentialsValidateFailedError:
            raise
        except requests.RequestException as ex:
            raise CredentialsValidateFailedError(
                "SophNet TTS credentials validation failed"
            ) from ex
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

    def get_tts_model_voices(
        self,
        model: str,
        credentials: dict,
        language: str | None = None,
    ) -> list[dict[str, str]] | None:
        del model, credentials
        voices = self.SUPPORTED_VOICES
        if language:
            voices = [item for item in voices if language in item.get("language", "")]
        return [{"name": item["name"], "value": item["mode"]} for item in voices]

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        del credentials
        return AIModelEntity(
            model=model,
            label=I18nObject(
                en_US=f"SophNet Text to Speech ({model})",
                zh_Hans=f"SophNet 文本转语音 ({model})",
            ),
            model_type=ModelType.TTS,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.DEFAULT_VOICE: self.DEFAULT_VOICE,
                ModelPropertyKey.VOICES: self.SUPPORTED_VOICES,
                ModelPropertyKey.AUDIO_TYPE: "mp3",
                ModelPropertyKey.WORD_LIMIT: 500,
                ModelPropertyKey.MAX_WORKERS: 3,
            },
            parameter_rules=[
                ParameterRule(
                    name="easyllm_id",
                    label=I18nObject(en_US="EasyLLM ID", zh_Hans="EasyLLM ID"),
                    type=ParameterType.STRING,
                    required=True,
                    default=model,
                ),
                ParameterRule(
                    name="streaming",
                    label=I18nObject(en_US="Use Streaming API", zh_Hans="使用流式接口"),
                    type=ParameterType.BOOLEAN,
                    required=False,
                    default=True,
                ),
                ParameterRule(
                    name="format",
                    label=I18nObject(en_US="Audio Format", zh_Hans="音频格式"),
                    type=ParameterType.STRING,
                    required=False,
                    default=self.DEFAULT_FORMAT,
                    options=self.SUPPORTED_FORMATS,
                ),
                ParameterRule(
                    name="volume",
                    label=I18nObject(en_US="Volume (0-100)", zh_Hans="音量 (0-100)"),
                    type=ParameterType.INT,
                    required=False,
                    default=50,
                ),
                ParameterRule(
                    name="speechRate",
                    label=I18nObject(en_US="Speech Rate (0.5-2)", zh_Hans="语速 (0.5-2)"),
                    type=ParameterType.FLOAT,
                    required=False,
                    default=1.0,
                ),
                ParameterRule(
                    name="pitchRate",
                    label=I18nObject(en_US="Pitch Rate (0.5-2)", zh_Hans="语调 (0.5-2)"),
                    type=ParameterType.FLOAT,
                    required=False,
                    default=1.0,
                ),
            ],
            pricing=PriceConfig(input=2.0, output=0.0, unit=0.0001, currency="RMB"),
        )

    def _build_request_params(
        self,
        credentials: Mapping,
        model: str,
        texts: list[str],
        voice: str,
        streaming: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        project_id = credentials.get("project_id")
        api_key = credentials.get("api_key")
        easyllm_id = credentials.get("easyllm_id") or model
        if not project_id:
            raise ValueError("project_id is required")
        if not api_key:
            raise ValueError("api_key is required")
        if not easyllm_id:
            raise ValueError("easyllm_id is required")

        operation = "synthesize-audio-stream" if streaming else "synthesize-audio"
        url = (
            f"https://www.sophnet.com/api/open-apis/projects/{project_id}"
            f"/easyllms/voice/{operation}"
        )
        synthesis_param: dict[str, Any] = {
            "model": credentials.get("model") or model or self.DEFAULT_MODEL,
            "voice": voice,
        }
        converters = {"volume": int, "speechRate": float, "pitchRate": float}
        if credentials.get("format"):
            synthesis_param["format"] = credentials["format"]
        for name, converter in converters.items():
            if credentials.get(name) is not None:
                synthesis_param[name] = converter(credentials[name])

        return (
            url,
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            {
                "easyllm_id": str(easyllm_id),
                "text": texts,
                "synthesis_param": synthesis_param,
            },
        )

    @staticmethod
    def _as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off", ""}:
                return False
        raise ValueError("streaming must be a boolean")

    @property
    def _invoke_error_mapping(self) -> dict:
        return {
            InvokeBadRequestError: [InvokeBadRequestError],
            InvokeConnectionError: [InvokeConnectionError],
            CredentialsValidateFailedError: [CredentialsValidateFailedError],
            InvokeError: [Exception],
        }
