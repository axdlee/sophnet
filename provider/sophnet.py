from collections.abc import Mapping

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType


class SophNetProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        model_instance = self.get_model_instance(ModelType.LLM)
        model_instance.validate_credentials(
            model="DeepSeek-V4-Flash",
            credentials=dict(credentials),
        )
