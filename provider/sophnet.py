import logging
from collections.abc import Mapping

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class SophNetProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        try:
            model_instance = self.get_model_instance(ModelType.LLM)
            model_instance.validate_credentials(
                model="DeepSeek-V3-Fast", credentials=credentials
            )
        except CredentialsValidateFailedError as ex:
            # 可选：logger.warning("Sophnet credentials invalid: %s", ex)
            raise CredentialsValidateFailedError(
                "必须在sophnet平台中完成以下操作: 控制台-> 项目详情 -> 添加DeepSeek-V3-Fast服务"
            ) from ex
        except Exception as ex:
            logger.exception(
                f"{self.get_provider_schema().provider} credentials validate failed"
            )
            raise ex
