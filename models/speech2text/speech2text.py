import json
import os
import tempfile
import time
import wave
from collections.abc import Mapping
from typing import IO, Any

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
from dify_plugin.interfaces.model.speech2text_model import Speech2TextModel


class SophnetSpeech2TextModel(Speech2TextModel):
    """SophNet asynchronous speech-to-text model."""

    def _invoke(
        self,
        model: str,
        credentials: dict,
        file: IO[bytes],
        user: str | None = None,
    ) -> str:
        del user
        url, headers, easyllm_id = self._build_request_params(credentials, model)
        files = self._multipart_files(file, easyllm_id)
        response: requests.Response | None = None
        try:
            response = requests.post(url, headers=headers, files=files, timeout=600)
            response.raise_for_status()
            task_data = response.json()
        except requests.RequestException as ex:
            raise InvokeConnectionError("SophNet speech-to-text request failed") from ex
        except ValueError as ex:
            raise InvokeBadRequestError("SophNet speech-to-text API returned invalid JSON") from ex
        finally:
            if response is not None:
                response.close()

        task_id = task_data.get("task_id") if isinstance(task_data, dict) else None
        if not task_id:
            raise InvokeBadRequestError("SophNet speech-to-text API did not return a task_id")
        return self._poll_task_result(credentials, str(task_id))

    def _poll_task_result(
        self,
        credentials: dict,
        task_id: str,
        max_retries: int = 60,
        interval: int = 5,
    ) -> str:
        project_id = credentials.get("project_id")
        api_key = credentials.get("api_key")
        if not project_id or not api_key:
            raise InvokeBadRequestError("project_id and api_key are required")

        url = (
            f"https://www.sophnet.com/api/open-apis/projects/{project_id}"
            f"/easyllms/speechtotext/transcriptions/{task_id}"
        )
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        current_interval = interval

        for retry in range(max_retries):
            data = self._get_task(url, headers)
            status = data.get("status")
            if status == "success":
                result = data.get("result")
                return result if isinstance(result, str) else ""
            if status == "failed":
                message = data.get("errorMsg") or "Unknown error"
                raise InvokeBadRequestError(f"SophNet transcription failed: {message}")
            if status not in {"waiting", "doing"}:
                raise InvokeBadRequestError(f"Unknown SophNet transcription status: {status}")

            if retry + 1 < max_retries:
                time.sleep(current_interval)
                if retry >= 10:
                    current_interval = min(current_interval + 1, 15)

        raise InvokeBadRequestError(
            f"SophNet transcription task {task_id} timed out after {max_retries} attempts"
        )

    @staticmethod
    def _get_task(url: str, headers: dict[str, str]) -> dict[str, Any]:
        response: requests.Response | None = None
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as ex:
            raise InvokeConnectionError("Failed to query SophNet transcription task") from ex
        except ValueError as ex:
            raise InvokeBadRequestError("SophNet transcription API returned invalid JSON") from ex
        finally:
            if response is not None:
                response.close()
        if not isinstance(data, dict):
            raise InvokeBadRequestError("SophNet transcription API returned an invalid response")
        return data

    def validate_credentials(self, model: str, credentials: Mapping) -> None:
        audio_file_path: str | None = None
        try:
            url, headers, easyllm_id = self._build_request_params(dict(credentials), model)
        except ValueError as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex
        try:
            audio_file_path = self._create_validation_audio()
            with open(audio_file_path, "rb") as audio_file:
                response = requests.post(
                    url,
                    headers=headers,
                    files=self._multipart_files(audio_file, easyllm_id),
                    timeout=30,
                )
                try:
                    response.raise_for_status()
                    data = response.json()
                finally:
                    response.close()
            if not isinstance(data, dict) or not data.get("task_id"):
                raise CredentialsValidateFailedError(
                    "SophNet speech-to-text validation did not return a task_id"
                )
        except CredentialsValidateFailedError:
            raise
        except requests.RequestException as ex:
            raise CredentialsValidateFailedError(
                "SophNet speech-to-text credentials validation request failed"
            ) from ex
        except Exception as ex:
            raise CredentialsValidateFailedError(
                "SophNet speech-to-text credentials validation failed"
            ) from ex
        finally:
            if audio_file_path:
                try:
                    os.unlink(audio_file_path)
                except FileNotFoundError:
                    pass

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        del credentials
        return AIModelEntity(
            model=model,
            label=I18nObject(
                en_US=f"SophNet Speech to Text ({model})",
                zh_Hans=f"SophNet 语音转文本 ({model})",
            ),
            model_type=ModelType.SPEECH2TEXT,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.FILE_UPLOAD_LIMIT: 1024,
                ModelPropertyKey.SUPPORTED_FILE_EXTENSIONS: (
                    "wav,mp3,m4a,flv,mp4,wma,3gp,amr,aac,ogg-opus,flac"
                ),
            },
            parameter_rules=[
                ParameterRule(
                    name="easyllm_id",
                    label=I18nObject(en_US="EasyLLM ID", zh_Hans="EasyLLM ID"),
                    type=ParameterType.STRING,
                    required=True,
                    default=model,
                )
            ],
            pricing=PriceConfig(input=0.0, output=0.0, unit=0.000001, currency="RMB"),
        )

    @staticmethod
    def _multipart_files(file: IO[bytes], easyllm_id: str) -> dict[str, Any]:
        return {
            "audio_file": file,
            "data": (
                None,
                json.dumps({"easyllm_id": easyllm_id}),
                "application/json",
            ),
        }

    @staticmethod
    def _build_request_params(credentials: Mapping, model: str) -> tuple[str, dict[str, str], str]:
        project_id = credentials.get("project_id")
        api_key = credentials.get("api_key")
        easyllm_id = credentials.get("easyllm_id") or model
        if not project_id:
            raise ValueError("project_id is required")
        if not api_key:
            raise ValueError("api_key is required")
        if not easyllm_id:
            raise ValueError("easyllm_id is required")

        url = (
            f"https://www.sophnet.com/api/open-apis/projects/{project_id}"
            "/easyllms/speechtotext/transcriptions"
        )
        return url, {"Authorization": f"Bearer {api_key}"}, str(easyllm_id)

    @staticmethod
    def _create_validation_audio() -> str:
        descriptor, path = tempfile.mkstemp(prefix="sophnet-stt-", suffix=".wav")
        os.close(descriptor)
        try:
            with wave.open(path, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\x00\x00" * 1600)
        except Exception:
            os.unlink(path)
            raise
        return path

    @property
    def _invoke_error_mapping(self) -> dict:
        return {
            InvokeBadRequestError: [InvokeBadRequestError],
            InvokeConnectionError: [InvokeConnectionError],
            CredentialsValidateFailedError: [CredentialsValidateFailedError],
            InvokeError: [Exception],
        }
