import requests
from typing import Optional
from dify_plugin.interfaces.model.openai_compatible.text_embedding import OAICompatEmbeddingModel
from dify_plugin.entities.model import EmbeddingInputType
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult


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
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Invoke text embedding model

        :param model: model name (easyllm_id)
        :param credentials: model credentials (must include project_id, api_key, dimensions)
        :param texts: texts to embed
        :param user: unique user id
        :param input_type: input type
        :return: embeddings result
        """
        self._add_custom_parameters(credentials)
        api_key = credentials["api_key"]
        dimensions = credentials.get("dimensions", 1024)
        easyllm_id = model
        
        endpoint_url = credentials.get("endpoint_url", "")
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"

        url = f"{endpoint_url}/embeddings"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "easyllm_id": easyllm_id,
            "input_texts": texts,
            "dimensions": dimensions
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Sophnet embedding API调用失败: {e}")

        # 解析返回结果，组装为TextEmbeddingResult
        embeddings = [item["embedding"] for item in data.get("data", [])]
        usage = data.get("usage", {})
        return TextEmbeddingResult(
            embeddings=embeddings,
            usage=usage
        )

    @classmethod
    def _add_custom_parameters(cls, credentials: dict) -> None:
        project_id = credentials.get("project_id")
        if not project_id:
            raise ValueError("project_id is required")
        api_key = credentials.get("api_key")
        if not api_key:
            raise ValueError("api_key is required")
        # endpoint_url参数可选，保留兼容性
        credentials["endpoint_url"] = f"https://www.sophnet.com/api/open-apis/projects/{project_id}/easyllms"
