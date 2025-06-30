import base64
import json
from typing import Dict, Generator, List, Mapping, Optional
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
    InvokeError,
)
from dify_plugin.interfaces.model.tts_model import TTSModel


class SophnetText2SpeechModel(TTSModel):
    """
    Model class for Sophnet Text to Speech model.
    
    支持流式和非流式接口：
    - 流式接口：https://www.sophnet.com/api/open-apis/projects/{ProjectId}/easyllms/voice/synthesize-audio-stream
    - 非流式接口：https://www.sophnet.com/api/open-apis/projects/{ProjectId}/easyllms/voice/synthesize-audio
    
    支持的音频格式：MP3、WAV等
    """

    # 默认音频格式
    DEFAULT_FORMAT = "MP3_16000HZ_MONO_128KBPS"
    # 默认模型
    DEFAULT_MODEL = "cosyvoice-v1"
    # 默认音色
    DEFAULT_VOICE = "longxiaochun"
    # 支持的音色列表
    SUPPORTED_VOICES = [
        {"name": "龙小春", "mode": "longxiaochun", "language": "zh"},
        {"name": "龙小夏", "mode": "longxiaoxia", "language": "zh"},
        {"name": "龙小成", "mode": "longxiaocheng", "language": "zh"},
        {"name": "龙小白", "mode": "longxiaobai", "language": "zh"},
        {"name": "龙老铁", "mode": "longlaotie", "language": "zh"},
        {"name": "龙叔", "mode": "longshu", "language": "zh"},
        {"name": "龙硕", "mode": "longshuo", "language": "zh"},
        {"name": "龙婧", "mode": "longjing", "language": "zh"},
        {"name": "龙悦", "mode": "longyue", "language": "zh"},
        {"name": "龙湾", "mode": "longwan", "language": "zh"},
        {"name": "龙成", "mode": "longcheng", "language": "zh"},
        {"name": "龙华", "mode": "longhua", "language": "zh"},
        {"name": "Stella", "mode": "loongstella", "language": "en"},
        {"name": "Bella", "mode": "loongbella", "language": "en"},
    ]
    # 支持的格式列表
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
        user: Optional[str] = None,
    ) -> Generator[bytes, None, None]:
        """
        调用文本转语音模型

        :param model: 模型名称
        :param tenant_id: 租户ID
        :param credentials: 模型凭证
        :param content_text: 要转换的文本内容
        :param voice: 音色
        :param user: 用户ID
        :return: 音频数据流
        """
        # 验证并获取有效的音色
        if not voice or voice not in [v["mode"] for v in self.SUPPORTED_VOICES]:
            voice = self.DEFAULT_VOICE

        # 获取流式/非流式设置
        streaming = credentials.get("streaming", True)
        
        # 根据文本长度拆分文本
        if len(content_text) > 500:
            sentences = self._split_text_into_sentences(content_text, max_length=500)
        else:
            sentences = [content_text]
        
        # 调用相应的接口
        if streaming:
            yield from self._invoke_streaming(model, credentials, sentences, voice)
        else:
            yield from self._invoke_non_streaming(model, credentials, sentences, voice)

    def _invoke_streaming(
        self, model: str, credentials: dict, sentences: List[str], voice: str
    ) -> Generator[bytes, None, None]:
        """
        调用流式文本转语音接口

        :param model: 模型名称
        :param credentials: 模型凭证
        :param sentences: 文本句子列表
        :param voice: 音色
        :return: 音频数据流
        """
        url, headers, payload = self._build_request_params(
            credentials, sentences, voice, True
        )
        
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            # 处理流式响应
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                    
                # 流式响应格式为: data: {...}
                if line.startswith('data:'):
                    try:
                        data = json.loads(line[5:])
                        if 'audioFrame' in data and data['audioFrame']:
                            # 解码base64音频数据
                            audio_bytes = base64.b64decode(data['audioFrame'])
                            yield audio_bytes
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            raise InvokeBadRequestError(f"Sophnet TTS streaming API call failed: {e}")

    def _invoke_non_streaming(
        self, model: str, credentials: dict, sentences: List[str], voice: str
    ) -> Generator[bytes, None, None]:
        """
        调用非流式文本转语音接口

        :param model: 模型名称
        :param credentials: 模型凭证
        :param sentences: 文本句子列表
        :param voice: 音色
        :return: 音频数据
        """
        url, headers, payload = self._build_request_params(
            credentials, sentences, voice, False
        )
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            # 非流式接口直接返回二进制音频数据
            yield response.content
            
        except Exception as e:
            raise InvokeBadRequestError(f"Sophnet TTS non-streaming API call failed: {e}")

    def validate_credentials(self, model: str, credentials: Mapping) -> None:
        """
        验证凭证有效性

        :param model: 模型名称
        :param credentials: 模型凭证
        """
        try:
            # 验证必要参数
            project_id = credentials.get("project_id")
            api_key = credentials.get("api_key")
            easyllm_id = credentials.get("easyllm_id", model)
            
            if not project_id:
                raise CredentialsValidateFailedError("project_id is required")
            if not api_key:
                raise CredentialsValidateFailedError("api_key is required")
            if not easyllm_id:
                raise CredentialsValidateFailedError("easyllm_id is required")
            
            # 使用简短的文本进行验证
            test_text = ["你好，这是一个测试"]
            
            # 构建请求参数
            url, headers, payload = self._build_request_params(
                credentials, test_text, self.DEFAULT_VOICE, False
            )
            
            # 发送请求验证
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed with status code {response.status_code}: {response.text}"
                )
                
        except Exception as ex:
            if isinstance(ex, CredentialsValidateFailedError):
                raise ex
            raise CredentialsValidateFailedError(str(ex)) from ex

    def get_tts_model_voices(self, model: str, credentials: dict, language: Optional[str] = None) -> Optional[List[Dict]]:
        """
        获取模型支持的音色列表

        :param model: 模型名称
        :param credentials: 模型凭证
        :param language: 语言
        :return: 音色列表
        """
        voices = self.SUPPORTED_VOICES
        
        if language:
            return [
                {"name": d["name"], "value": d["mode"]}
                for d in voices
                if language and language in d.get("language", "")
            ]
        else:
            return [{"name": d["name"], "value": d["mode"]} for d in voices]

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        """
        提供可定制的模型配置，支持project_id、api_key、easyllm_id等参数配置
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=f"Sophnet Text to Speech ({model})", zh_Hans=f"Sophnet 文本转语音 ({model})"),
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
                    default=model
                ),
                ParameterRule(
                    name="streaming",
                    label=I18nObject(en_US="Use Streaming API", zh_Hans="使用流式接口"),
                    type=ParameterType.BOOLEAN,
                    required=False,
                    default="true"
                ),
                ParameterRule(
                    name="format",
                    label=I18nObject(en_US="Audio Format", zh_Hans="音频格式"),
                    type=ParameterType.STRING,
                    required=False,
                    default=self.DEFAULT_FORMAT,
                    options=self.SUPPORTED_FORMATS
                ),
                ParameterRule(
                    name="volume",
                    label=I18nObject(en_US="Volume (0-100)", zh_Hans="音量 (0-100)"),
                    type=ParameterType.INT,
                    required=False,
                    default="50"
                ),
                ParameterRule(
                    name="speechRate",
                    label=I18nObject(en_US="Speech Rate (0.5-2)", zh_Hans="语速 (0.5-2)"),
                    type=ParameterType.FLOAT,
                    required=False,
                    default="1.0"
                ),
                ParameterRule(
                    name="pitchRate",
                    label=I18nObject(en_US="Pitch Rate (0.5-2)", zh_Hans="语调 (0.5-2)"),
                    type=ParameterType.FLOAT,
                    required=False,
                    default="1.0"
                ),
            ],
            pricing=PriceConfig(
                input=2.0, # 2元/万字符
                output=0.0, # 暂不收费，后续根据实际使用情况调整
                unit=0.0001,
                currency="RMB"
            ),
        )
        return entity

    def _build_request_params(
        self, credentials: dict, texts: List[str], voice: str, streaming: bool
    ) -> tuple:
        """
        构建请求参数

        :param credentials: 凭证信息
        :param texts: 文本列表
        :param voice: 音色
        :param streaming: 是否使用流式接口
        :return: url, headers, payload
        """
        project_id = credentials.get("project_id")
        if not project_id:
            raise ValueError("project_id is required")
        
        api_key = credentials.get("api_key")
        if not api_key:
            raise ValueError("api_key is required")
        
        easyllm_id = credentials.get("easyllm_id")
        if not easyllm_id:
            raise ValueError("easyllm_id is required")
        
        # 构建URL
        if streaming:
            url = f"https://www.sophnet.com/api/open-apis/projects/{project_id}/easyllms/voice/synthesize-audio-stream"
        else:
            url = f"https://www.sophnet.com/api/open-apis/projects/{project_id}/easyllms/voice/synthesize-audio"
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体
        synthesis_param = {
            "model": credentials.get("model", self.DEFAULT_MODEL),
            "voice": voice,
        }
        
        # 添加可选参数
        if "format" in credentials:
            synthesis_param["format"] = credentials["format"]
        
        if "volume" in credentials:
            synthesis_param["volume"] = int(credentials["volume"])
        
        if "speechRate" in credentials:
            synthesis_param["speechRate"] = float(credentials["speechRate"])
        
        if "pitchRate" in credentials:
            synthesis_param["pitchRate"] = float(credentials["pitchRate"])
        
        payload = {
            "easyllm_id": easyllm_id,
            "text": texts,
            "synthesis_param": synthesis_param
        }
        
        return url, headers, payload
    
    @property
    def _invoke_error_mapping(self) -> dict:
        return {
            InvokeBadRequestError: [InvokeBadRequestError],
            CredentialsValidateFailedError: [CredentialsValidateFailedError],
            InvokeError: [Exception],
        }
