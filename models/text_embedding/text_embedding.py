import json
import time
from typing import Optional, List
from dify_plugin.interfaces.model.openai_compatible.text_embedding import OAICompatEmbeddingModel
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
from dify_plugin.entities.model.text_embedding import (
    EmbeddingUsage,
    TextEmbeddingResult,
)
from dify_plugin.errors.model import CredentialsValidateFailedError
import requests

class SophnetTextEmbeddingModel(OAICompatEmbeddingModel):
    """
    Model class for Sophnet text embedding model.
    """

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._add_custom_parameters(credentials)
        super().validate_credentials(model, credentials)

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: List[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Sophnet embedding接口分批调用，兼容OAI接口风格
        """
        self._add_custom_parameters(credentials)
        api_key = credentials["api_key"]
        dimensions = int(credentials.get("dimensions", 1024))
        easyllm_id = credentials.get("easyllm_id", model)
        endpoint_url = credentials.get("endpoint_url", "")
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"
        url = f"{endpoint_url}embeddings"

        max_batch = 10  # Sophnet API最大支持10条
        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(texts), max_batch):
            batch_texts = texts[i:i+max_batch]
            payload = {
                "easyllm_id": easyllm_id,
                "input_texts": batch_texts,
                "dimensions": dimensions
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                raise RuntimeError(f"Sophnet embedding API调用失败: {e}")

            # 解析返回结果
            batch_embeddings = [item["embedding"] for item in data.get("data", [])]
            all_embeddings.extend(batch_embeddings)
            usage = data.get("usage", {})
            total_tokens += usage.get("total_tokens", 0) or 0

        # 用法统计
        usage_obj = self._calc_response_usage(model=model, credentials=credentials, tokens=total_tokens)

        return TextEmbeddingResult(
            embeddings=all_embeddings,
            usage=usage_obj,
            model=model
        )

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        """
        支持easyllm_id和dimensions为可配置参数，便于前端/调用方动态配置
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=f"Sophnet Embedding ({model})"),
            model_type=ModelType.TEXT_EMBEDDING,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(credentials.get("context_size", 8192)),
                ModelPropertyKey.MAX_CHUNKS: 10,
            },
            parameter_rules=[
                ParameterRule(
                    name="easyllm_id",
                    label=I18nObject(en_US="EasyLLM ID", zh_Hans="EasyLLM ID"),
                    type=ParameterType.STRING,
                    required=True,
                    default=model
                ),
                ParameterRule(
                    name="dimensions",
                    label=I18nObject(en_US="Embedding Dimensions, support 1,024/768/512/256/128/64", zh_Hans="输出Embeddings的维度，支持1,024/768/512/256/128/64"),
                    type=ParameterType.INTEGER,
                    required=True,
                    default=1024,
                    options=[1024, 768, 512, 256, 128, 64]
                ),
            ],
            pricing=PriceConfig(
                input=0.5, # https://www.sophnet.com/docs/component/fee.html
                output=0,
                unit=0.000001,
                currency="RMB"
            ),
        )
        return entity

    @classmethod
    def _add_custom_parameters(cls, credentials: dict) -> None:
        project_id = credentials.get("project_id")
        if not project_id:
            raise ValueError("project_id is required")
        api_key = credentials.get("api_key")
        if not api_key:
            raise ValueError("api_key is required")
        credentials["endpoint_url"] = f"https://www.sophnet.com/api/open-apis/projects/{project_id}/easyllms"
