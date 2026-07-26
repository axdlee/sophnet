from collections.abc import Generator

from dify_plugin import OAICompatLargeLanguageModel
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
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    UserPromptMessage,
    VideoPromptMessageContent,
)


class SophnetLargeLanguageModel(OAICompatLargeLanguageModel):
    """SophNet OpenAI-compatible large language model."""

    endpoint_url = "https://www.sophnet.com/api/open-apis/v1"
    MODEL_ALIASES = {
        "DeepSeek-V3.1": "DeepSeek-V4-Flash",
        "DeepSeek-R1": "DeepSeek-V4-Pro",
        "qwen-vl-plus-latest": "qwen3-vl-plus",
    }

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: list[PromptMessageTool] | None = None,
        stop: list[str] | None = None,
        stream: bool = True,
        user: str | None = None,
    ) -> LLMResult | Generator:
        resolved_model = self._resolve_model(model)
        request_credentials = dict(credentials)
        self._add_custom_parameters(request_credentials)
        self._add_function_call(resolved_model, request_credentials)
        return super()._invoke(
            resolved_model,
            request_credentials,
            prompt_messages,
            model_parameters,
            tools,
            stop,
            stream,
            user,
        )

    def _convert_prompt_message_to_dict(
        self,
        message: PromptMessage,
        credentials: dict | None = None,
    ) -> dict:
        """Preserve video parts omitted by dify_plugin 0.10's OAI converter."""
        converted = super()._convert_prompt_message_to_dict(message, credentials)
        if not isinstance(message, UserPromptMessage) or not isinstance(message.content, list):
            return converted

        converted_parts = iter(converted.get("content", []))
        content_parts: list[dict] = []
        for part in message.content:
            if isinstance(part, VideoPromptMessageContent):
                content_parts.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": part.data},
                    }
                )
            elif part.type in {
                PromptMessageContentType.TEXT,
                PromptMessageContentType.IMAGE,
            }:
                content_parts.append(next(converted_parts))

        content_parts.extend(converted_parts)
        converted["content"] = content_parts
        return converted

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: list[PromptMessageTool] | None = None,
    ) -> int:
        return super().get_num_tokens(
            self._resolve_model(model), credentials, prompt_messages, tools
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        request_credentials = dict(credentials)
        self._add_custom_parameters(request_credentials)
        super().validate_credentials(self._resolve_model(model), request_credentials)

    @classmethod
    def _resolve_model(cls, model: str) -> str:
        return cls.MODEL_ALIASES.get(model, model)

    @classmethod
    def _add_custom_parameters(cls, credentials: dict) -> None:
        credentials["mode"] = LLMMode.CHAT.value
        credentials["endpoint_url"] = cls.endpoint_url

    def _add_function_call(self, model: str, credentials: dict) -> None:
        model_schema = self.get_model_schema(model, credentials)
        supports_tools = model_schema and {
            ModelFeature.TOOL_CALL,
            ModelFeature.MULTI_TOOL_CALL,
        }.intersection(model_schema.features or [])
        if supports_tools:
            credentials["function_calling_type"] = "tool_call"
            credentials["stream_function_calling"] = "supported"
        else:
            credentials.pop("function_calling_type", None)
            credentials.pop("stream_function_calling", None)

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        features: list[ModelFeature] = []
        function_calling_type = credentials.get("function_calling_type")
        if function_calling_type in {"function_call", "tool_call"}:
            features.extend(
                [
                    ModelFeature.TOOL_CALL,
                    ModelFeature.AGENT_THOUGHT,
                    ModelFeature.STREAM_TOOL_CALL,
                ]
            )
        if self._credential_enabled(credentials.get("vision")):
            features.append(ModelFeature.VISION)
        if self._credential_enabled(credentials.get("video")):
            features.append(ModelFeature.VIDEO)
        if self._credential_enabled(credentials.get("structured_output")):
            features.append(ModelFeature.STRUCTURED_OUTPUT)

        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model, zh_Hans=model),
            model_type=ModelType.LLM,
            features=features,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(credentials.get("context_size", 65536)),
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
            },
            parameter_rules=self._parameter_rules(credentials),
        )

    @staticmethod
    def _credential_enabled(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "on"}

    @staticmethod
    def _parameter_rules(credentials: dict) -> list[ParameterRule]:
        return [
            ParameterRule(
                name="temperature",
                use_template="temperature",
                label=I18nObject(en_US="Temperature", zh_Hans="温度"),
                type=ParameterType.FLOAT,
            ),
            ParameterRule(
                name="max_tokens",
                use_template="max_tokens",
                default=min(8192, int(credentials.get("max_tokens", 65536))),
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
                name="frequency_penalty",
                use_template="frequency_penalty",
                label=I18nObject(en_US="Frequency Penalty", zh_Hans="重复惩罚"),
                type=ParameterType.FLOAT,
            ),
            ParameterRule(
                name="presence_penalty",
                use_template="presence_penalty",
                label=I18nObject(en_US="Presence Penalty", zh_Hans="存在惩罚"),
                type=ParameterType.FLOAT,
            ),
            ParameterRule(
                name="response_format",
                use_template="response_format",
                label=I18nObject(en_US="Response Format", zh_Hans="响应格式"),
                type=ParameterType.STRING,
                required=False,
                options=["text", "json_object"],
            ),
        ]
