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
        # 获取模型参数
        context_size = self._get_context_size(model, credentials)
        max_chunks = self._get_max_chunks(model, credentials)
        url, headers, easyllm_id, dimensions = self._build_request_params(credentials, model)

        # 1. 按context_size截断每条文本
        processed_texts = []
        for text in texts:
            num_tokens = self._get_num_tokens_by_gpt2(text)
            if num_tokens > context_size:
                # 近似按字符截断
                cutoff = int(len(text) * context_size / num_tokens)
                processed_texts.append(text[:cutoff])
            else:
                processed_texts.append(text)

        # 2. 按max_chunks分批请求
        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(processed_texts), max_chunks):
            batch_texts = processed_texts[i:i+max_chunks]
            payload = {
                "easyllm_id": easyllm_id,
                "input_texts": batch_texts,
                "dimensions": dimensions
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                raise RuntimeError(f"Sophnet embedding API call failed: {e}")

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
    

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        校验Sophnet API凭证有效性，需project_id、api_key、easyllm_id、dimensions
        """
        url, headers, easyllm_id, dimensions = self._build_request_params(credentials, model)
        payload = {
            "easyllm_id": easyllm_id,
            "input_texts": ["ping"],
            "dimensions": dimensions
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed with status code {response.status_code}"
                )
            data = response.json()
            if not data.get("data"):
                raise CredentialsValidateFailedError("Credentials validation failed: no embedding data returned")
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

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
                ModelPropertyKey.CONTEXT_SIZE: 8192,
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


    def _build_request_params(self, credentials: dict, model: str):
        """
        提取公共参数和请求头，返回url, headers, easyllm_id, dimensions
        """
        self._add_custom_parameters(credentials)
        api_key = credentials.get("api_key")
        easyllm_id = credentials.get("easyllm_id", model)
        dimensions = int(credentials.get("dimensions", 1024))
        endpoint_url = credentials.get("endpoint_url", "")
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"
        url = f"{endpoint_url}embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        return url, headers, easyllm_id, dimensions
    
    @classmethod
    def _add_custom_parameters(cls, credentials: dict) -> None:
        project_id = credentials.get("project_id")
        if not project_id:
            raise ValueError("project_id is required")
        api_key = credentials.get("api_key")
        if not api_key:
            raise ValueError("api_key is required")
        credentials["endpoint_url"] = f"https://www.sophnet.com/api/open-apis/projects/{project_id}/easyllms"
