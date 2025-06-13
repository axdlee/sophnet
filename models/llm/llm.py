import logging
from collections.abc import Generator
from typing import Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
)

from dify_plugin.entities.model.llm import (
    LLMMode,
    LLMResult,
)
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
)

logger = logging.getLogger(__name__)


class SophnetLargeLanguageModel(OAICompatLargeLanguageModel):
    """
    Model class for sophnet large language model.
    """

    # 设置默认的endpoint_url
    endpoint_url = "https://www.sophnet.com/api/open-apis/v1"

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        self._add_custom_parameters(credentials)
        self._add_function_call(model, credentials)
        return super()._invoke(
            model, credentials, prompt_messages, model_parameters, tools, stop, stream
        )
   
    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:
        """
        return 0

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            self._add_custom_parameters(credentials)
            super().validate_credentials(model, credentials)
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @classmethod
    def _add_custom_parameters(cls, credentials: dict) -> None:
        credentials["mode"] = "chat"
        credentials["endpoint_url"] = cls.endpoint_url

    def _add_function_call(self, model: str, credentials: dict) -> None:
        model_schema = self.get_model_schema(model, credentials)
        if model_schema and {ModelFeature.TOOL_CALL, ModelFeature.MULTI_TOOL_CALL}.intersection(
            model_schema.features or []
        ):
            credentials["function_calling_type"] = "tool_call"    

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        If your model supports fine-tuning, this method returns the schema of the base model
        but renamed to the fine-tuned model name.

        :param model: model name
        :param credentials: credentials

        :return: model schema
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model, zh_Hans=model),
            model_type=ModelType.LLM,
            features=(
                [
                    ModelFeature.TOOL_CALL,
                    ModelFeature.MULTI_TOOL_CALL,
                    ModelFeature.STREAM_TOOL_CALL,
                ]
                if credentials.get("function_calling_type") == "tool_call"
                else []
            ),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(
                    credentials.get("context_size", 64000)
                ),
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
            },
            parameter_rules=[
                ParameterRule(
                    name="temperature",
                    use_template="temperature",
                    label=I18nObject(en_US="Temperature", zh_Hans="温度"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="max_tokens",
                    use_template="max_tokens",
                    default=16384,
                    min=1,
                    max=int(credentials.get("max_tokens", 65536)),
                    label=I18nObject(en_US="Max Tokens", zh_Hans="最大标记"),
                    type=ParameterType.INT,
                ),
                ParameterRule(
                    name="top_p",
                    use_template="top_p",
                    label=I18nObject(en_US="Top P", zh_Hans="Top P"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="top_k",
                    use_template="top_k",
                    label=I18nObject(en_US="Top K", zh_Hans="Top K"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="frequency_penalty",
                    use_template="frequency_penalty",
                    label=I18nObject(en_US="Frequency Penalty", zh_Hans="重复惩罚"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="enable_thinking",
                    use_template="enable_thinking",
                    default=True,
                    label=I18nObject(en_US="Thinking mode", zh_Hans="启用思考模式"),
                    type=ParameterType.BOOLEAN,
                ),
                ParameterRule(
                    name="thinking_budget",
                    use_template="thinking_budget",
                    default=2048,
                    min=1,
                    max=int(credentials.get("thinking_budget", 16384)),
                    label=I18nObject(en_US="Thinking budget", zh_Hans="思考长度限制"),
                    type=ParameterType.INT,
                ),
            ],
        )

        return entity
