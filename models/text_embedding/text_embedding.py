from typing import Any

import requests
from dify_plugin import OAICompatEmbeddingModel
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import (
    AIModelEntity,
    EmbeddingInputType,
    FetchFrom,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
    PriceConfig,
)
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
    InvokeConnectionError,
)


class SophnetTextEmbeddingModel(OAICompatEmbeddingModel):
    """SophNet text embedding model using the EasyLLM project API."""

    _VARIABLE_DIMENSIONS = frozenset({1024, 768, 512, 256, 128, 64})

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: str | None = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        del user, input_type
        context_size = self._get_context_size(model, credentials)
        max_chunks = self._get_max_chunks(model, credentials)
        url, headers, easyllm_id, dimensions = self._build_request_params(credentials, model)
        processed_texts = [self._truncate_text(text, context_size) for text in texts]

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        for offset in range(0, len(processed_texts), max_chunks):
            batch_texts = processed_texts[offset : offset + max_chunks]
            data = self._request_embeddings(
                url,
                headers,
                {"easyllm_id": easyllm_id, "input_texts": batch_texts, "dimensions": dimensions},
            )
            batch_embeddings = self._extract_embeddings(data, len(batch_texts), dimensions)
            all_embeddings.extend(batch_embeddings)
            total_tokens += int(data.get("usage", {}).get("total_tokens") or 0)

        return TextEmbeddingResult(
            embeddings=all_embeddings,
            usage=self._calc_response_usage(
                model=model, credentials=credentials, tokens=total_tokens
            ),
            model=model,
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            url, headers, easyllm_id, dimensions = self._build_request_params(credentials, model)
            data = self._request_embeddings(
                url,
                headers,
                {"easyllm_id": easyllm_id, "input_texts": ["ping"], "dimensions": dimensions},
                timeout=10,
            )
            self._extract_embeddings(data, 1, dimensions)
        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            raise CredentialsValidateFailedError(
                "SophNet embedding credentials validation failed"
            ) from ex

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        return AIModelEntity(
            model=model,
            label=I18nObject(
                en_US=f"SophNet Embedding ({model})",
                zh_Hans=f"SophNet 向量模型 ({model})",
            ),
            model_type=ModelType.TEXT_EMBEDDING,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: 8192,
                ModelPropertyKey.MAX_CHUNKS: 10,
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
                    name="dimensions",
                    label=I18nObject(
                        en_US="Embedding dimensions",
                        zh_Hans="向量维度",
                    ),
                    type=ParameterType.INT,
                    required=True,
                    default=1024,
                    options=["1024", "768", "512", "256", "128", "64"],
                ),
            ],
            pricing=PriceConfig(input=0.5, output=0, unit=0.000001, currency="RMB"),
        )

    def _truncate_text(self, text: str, context_size: int) -> str:
        token_count = self._get_num_tokens_by_gpt2(text)
        if token_count <= context_size:
            return text
        cutoff = max(1, int(len(text) * context_size / token_count))
        return text[:cutoff]

    @staticmethod
    def _request_embeddings(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int = 30,
    ) -> dict[str, Any]:
        response: requests.Response | None = None
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as ex:
            raise InvokeConnectionError("SophNet embedding API request failed") from ex
        except ValueError as ex:
            raise InvokeBadRequestError("SophNet embedding API returned invalid JSON") from ex
        finally:
            if response is not None:
                response.close()
        if not isinstance(data, dict):
            raise InvokeBadRequestError("SophNet embedding API returned an invalid response")
        return data

    @staticmethod
    def _extract_embeddings(
        data: dict[str, Any], expected_count: int, expected_dimensions: int | None = None
    ) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list) or len(items) != expected_count:
            raise InvokeBadRequestError(
                f"SophNet embedding API returned {len(items) if isinstance(items, list) else 0} "
                f"vectors for {expected_count} inputs"
            )
        try:
            embeddings = [item["embedding"] for item in items]
        except (KeyError, TypeError) as ex:
            raise InvokeBadRequestError(
                "SophNet embedding response is missing embedding data"
            ) from ex
        for embedding in embeddings:
            if not isinstance(embedding, list) or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in embedding
            ):
                raise InvokeBadRequestError("SophNet embedding response contains an invalid vector")
            if expected_dimensions is not None and len(embedding) != expected_dimensions:
                raise InvokeBadRequestError(
                    f"SophNet embedding API returned a {len(embedding)}-dimension vector; "
                    f"expected {expected_dimensions}"
                )
        return embeddings

    def _build_request_params(
        self, credentials: dict, model: str
    ) -> tuple[str, dict[str, str], str, int]:
        project_id = credentials.get("project_id")
        api_key = credentials.get("api_key")
        if not project_id:
            raise ValueError("project_id is required")
        if not api_key:
            raise ValueError("api_key is required")

        easyllm_id = str(credentials.get("easyllm_id") or model)
        dimensions = self._validate_dimensions(model, int(credentials.get("dimensions", 1024)))
        url = f"https://www.sophnet.com/api/open-apis/projects/{project_id}/easyllms/embeddings"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        return url, headers, easyllm_id, dimensions

    @classmethod
    def _validate_dimensions(cls, model: str, dimensions: int) -> int:
        if model == "bge-m3" and dimensions != 1024:
            raise ValueError("bge-m3 only supports 1024 dimensions")
        if model != "bge-m3" and dimensions not in cls._VARIABLE_DIMENSIONS:
            supported = ", ".join(
                str(item) for item in sorted(cls._VARIABLE_DIMENSIONS, reverse=True)
            )
            raise ValueError(
                f"unsupported embedding dimensions: {dimensions}; expected one of {supported}"
            )
        return dimensions
