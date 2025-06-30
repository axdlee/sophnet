import time
from typing import IO, Optional
import requests

from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
    InvokeError,
)
from dify_plugin.interfaces.model.speech2text_model import Speech2TextModel
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


class SophnetSpeech2TextModel(Speech2TextModel):
    """
    Model class for Sophnet Speech to text model.
    
    支持音频格式：wav、mp3、m4a、flv、mp4、wma、3gp、amr、aac、ogg-opus、flac
    音频限制：音频时长不能大于5小时，文件大小不超过1GB
    识别有效时间：识别结果在服务端保存24小时
    """

    def _invoke(self, model: str, credentials: dict, file: IO[bytes], user: Optional[str] = None) -> str:
        """
        Invoke speech2text model

        :param model: model name
        :param credentials: model credentials
        :param file: audio file
        :param user: unique user id
        :return: text for given audio file
        """
        # 构建请求参数
        url, headers, easyllm_id = self._build_request_params(credentials, model)
        
        # 准备表单数据
        files = {
            'audio_file': file,
            'data': (None, f'{{"easyllm_id": "{easyllm_id}"}}', 'application/json')
        }
        
        # 设置multipart/form-data请求头
        request_headers = headers.copy()
        request_headers['Content-Type'] = 'multipart/form-data'
        
        # 创建转录任务
        try:
            response = requests.post(url, headers=headers, files=files, timeout=600)
            response.raise_for_status()
            task_data = response.json()
            
            if 'taskId' not in task_data:
                raise InvokeBadRequestError(f"Failed to create transcription task: {response.text}")
            
            task_id = task_data['taskId']
            
            # 轮询获取结果
            return self._poll_task_result(credentials, task_id)
            
        except Exception as e:
            raise InvokeBadRequestError(f"Sophnet speech2text API call failed: {e}")

    def _poll_task_result(self, credentials: dict, task_id: str, max_retries: int = 60, interval: int = 5) -> str:
        """
        轮询获取转录任务结果
        
        任务状态说明：
        - waiting: 任务等待中
        - doing: 任务执行中
        - success: 任务成功
        - failed: 任务失败
        
        :param credentials: 凭证信息
        :param task_id: 任务ID
        :param max_retries: 最大重试次数
        :param interval: 轮询间隔(秒)
        :return: 转录文本结果
        """
        project_id = credentials.get("project_id")
        api_key = credentials.get("api_key")
        
        url = f"https://www.sophnet.com/api/open-apis/projects/{project_id}/easyllms/speechtotext/transcriptions/{task_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        for retry in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                status = data.get("status", "")
                
                if status == "success":
                    return data.get("result", "")
                elif status == "failed":
                    error_msg = data.get("errorMsg", "Unknown error")
                    raise InvokeBadRequestError(f"Transcription failed: {error_msg}")
                elif status == "waiting":
                    # 任务等待中，继续等待
                    pass
                elif status == "doing":
                    # 任务执行中，继续等待
                    pass
                else:
                    # 未知状态
                    raise InvokeBadRequestError(f"Unknown task status: {status}")
                
                # 如果任务仍在进行中，等待后重试
                time.sleep(interval)
                
                # 对于较长音频，适当增加等待时间
                if retry > 10 and interval < 15:
                    interval = min(interval + 1, 15)  # 逐渐增加等待时间，但不超过15秒
                
            except Exception as e:
                if isinstance(e, InvokeBadRequestError):
                    raise e
                raise InvokeBadRequestError(f"Failed to get transcription result: {e}")
        
        raise InvokeBadRequestError(f"Transcription task {task_id} timed out after {max_retries * interval} seconds")

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        验证凭证有效性
        
        :param model: 模型名称
        :param credentials: 凭证信息
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
            
            # 使用简短的音频文件进行验证
            audio_file_path = self._get_demo_file_path()
            
            with open(audio_file_path, "rb") as audio_file:
                # 尝试调用API但不等待完整结果
                url, headers, _ = self._build_request_params(credentials, model)
                files = {
                    'audio_file': audio_file,
                    'data': (None, f'{{"easyllm_id": "{easyllm_id}"}}', 'application/json')
                }
                
                response = requests.post(url, headers=headers, files=files, timeout=30)
                if response.status_code != 200:
                    raise CredentialsValidateFailedError(
                        f"Credentials validation failed with status code {response.status_code}: {response.text}"
                    )
                
                # 检查返回的任务ID
                data = response.json()
                if 'taskId' not in data:
                    raise CredentialsValidateFailedError("Invalid API response: no taskId returned")
                
        except Exception as ex:
            if isinstance(ex, CredentialsValidateFailedError):
                raise ex
            raise CredentialsValidateFailedError(str(ex)) from ex
    
    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        """
        提供可定制的模型配置，支持project_id、api_key、easyllm_id等参数配置
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=f"Sophnet Speech to Text ({model})", zh_Hans=f"Sophnet 语音转文本 ({model})"),
            model_type=ModelType.SPEECH2TEXT,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.FILE_UPLOAD_LIMIT: 1024,  # 1GB 限制
                ModelPropertyKey.SUPPORTED_FILE_EXTENSIONS: "wav,mp3,m4a,flv,mp4,wma,3gp,amr,aac,ogg-opus,flac",
            },
            parameter_rules=[
                ParameterRule(
                    name="easyllm_id",
                    label=I18nObject(en_US="EasyLLM ID", zh_Hans="EasyLLM ID"),
                    type=ParameterType.STRING,
                    required=True,
                    default=model
                ),
            ],
            pricing=PriceConfig(
                input=0.0,  # 1.2元/小时
                output=0.0, # 暂不收费，后续根据实际使用情况调整
                unit=0.000001,
                currency="RMB"
            ),
        )
        return entity

    def _build_request_params(self, credentials: dict, model: str):
        """
        构建请求参数
        
        :param credentials: 凭证信息
        :param model: 模型名称
        :return: url, headers, easyllm_id
        """
        project_id = credentials.get("project_id")
        if not project_id:
            raise ValueError("project_id is required")
        
        api_key = credentials.get("api_key")
        if not api_key:
            raise ValueError("api_key is required")
        
        easyllm_id = credentials.get("easyllm_id", model)
        
        url = f"https://www.sophnet.com/api/open-apis/projects/{project_id}/easyllms/speechtotext/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        return url, headers, easyllm_id

    @property
    def _invoke_error_mapping(self) -> dict:
        # 只需简单映射即可
        return {
            InvokeBadRequestError: [InvokeBadRequestError],
            CredentialsValidateFailedError: [CredentialsValidateFailedError],
            InvokeError: [Exception],
        }
