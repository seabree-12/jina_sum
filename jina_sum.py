# encoding:utf-8
import json
import os
import html
from urllib.parse import urlparse

import requests
import dashscope
from dashscope import Generation

import plugins
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from plugins import *

@plugins.register(
    name="JinaSum",
    desire_priority=10,
    hidden=False,
    desc="Sum url link content with jina reader and LLM models (OpenAI/Tongyi)",
    version="0.0.2",
    author="seabree-12",
)
class JinaSum(Plugin):

    jina_reader_base = "https://r.jina.ai"
    open_ai_api_base = "https://api.openai.com/v1"
    open_ai_model = "gpt-3.5-turbo"
    dashscope_model = "qwen-max"
    preferred_api = "openai"  # 默认使用openai
    max_words = 8000
    prompt = "我需要对下面引号内文档进行总结，总结输出包括以下三个部分：\n📖 一句话总结\n🔑 关键要点,用数字序号列出3-5个文章的核心内容\n🏷 标签: #xx #xx\n请使用emoji让你的表达更生动\n\n"
    white_url_list = []
    black_url_list = [
        "https://support.weixin.qq.com", # 视频号视频
        "https://channels-aladin.wxqcloud.qq.com", # 视频号音乐
    ]

    def __init__(self):
        super().__init__()
        try:
            self.config = super().load_config()
            if not self.config:
                self.config = self._load_config_template()
            
            # 基础配置
            self.jina_reader_base = self.config.get("jina_reader_base", self.jina_reader_base)
            self.max_words = self.config.get("max_words", self.max_words)
            self.prompt = self.config.get("prompt", self.prompt)
            self.white_url_list = self.config.get("white_url_list", self.white_url_list)
            self.black_url_list = self.config.get("black_url_list", self.black_url_list)
            
            # OpenAI配置
            self.open_ai_api_base = self.config.get("open_ai_api_base", self.open_ai_api_base)
            self.open_ai_api_key = self.config.get("open_ai_api_key", "")
            self.open_ai_model = self.config.get("open_ai_model", self.open_ai_model)
            
            # DashScope配置
            self.dashscope_api_key = self.config.get("dashscope_api_key", "")
            self.dashscope_model = self.config.get("dashscope_model", self.dashscope_model)
            if self.dashscope_api_key:
                dashscope.api_key = self.dashscope_api_key
            
            # API选择配置
            self.preferred_api = self.config.get("preferred_api", self.preferred_api)
            
            # 验证API可用性
            self.available_apis = []
            if self.open_ai_api_key:
                self.available_apis.append("openai")
            if self.dashscope_api_key:
                self.available_apis.append("dashscope")
            
            if not self.available_apis:
                raise Exception("No available API keys configured")
            
            # 如果首选API未配置，使用第一个可用的API
            if self.preferred_api not in self.available_apis:
                self.preferred_api = self.available_apis[0]
                
            logger.info(f"[JinaSum] inited, preferred_api={self.preferred_api}, available_apis={self.available_apis}")
            self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        except Exception as e:
            logger.error(f"[JinaSum] 初始化异常：{e}")
            raise "[JinaSum] init failed, ignore "

    def on_handle_context(self, e_context: EventContext, retry_count: int = 0):
        try:
            context = e_context["context"]
            content = context.content
            if context.type != ContextType.SHARING and context.type != ContextType.TEXT:
                return
            if not self._check_url(content):
                logger.debug(f"[JinaSum] {content} is not a valid url, skip")
                return
            if retry_count == 0:
                logger.debug("[JinaSum] on_handle_context. content: %s" % content)
                reply = Reply(ReplyType.TEXT, "🎉正在为您生成总结，请稍候...")
                channel = e_context["channel"]
                channel.send(reply, context)

            target_url = html.unescape(content)
            jina_url = self._get_jina_url(target_url)
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
            response = requests.get(jina_url, headers=headers, timeout=60)
            response.raise_for_status()
            target_url_content = response.text[:self.max_words]
            
            # 根据配置选择使用的API
            if self.preferred_api == "openai" and "openai" in self.available_apis:
                result = self._summarize_with_openai(target_url_content)
            else:
                result = self._summarize_with_dashscope(target_url_content)
            
            reply = Reply(ReplyType.TEXT, result)
            e_context["reply"] = reply
            e_context.action = EventAction.BREAK_PASS

        except Exception as e:
            if retry_count < 3:
                logger.warning(f"[JinaSum] {str(e)}, retry {retry_count + 1}")
                self.on_handle_context(e_context, retry_count + 1)
                return

            logger.exception(f"[JinaSum] {str(e)}")
            reply = Reply(ReplyType.ERROR, "我暂时无法总结链接，请稍后再试")
            e_context["reply"] = reply
            e_context.action = EventAction.BREAK_PASS

    def _summarize_with_openai(self, content: str) -> str:
        """使用OpenAI API进行总结"""
        openai_chat_url = self._get_openai_chat_url()
        openai_headers = self._get_openai_headers()
        openai_payload = self._get_openai_payload(content)
        
        response = requests.post(openai_chat_url, headers=openai_headers, json=openai_payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _summarize_with_dashscope(self, content: str) -> str:
        """使用DashScope API进行总结"""
        sum_prompt = f"{self.prompt}\n\n'''{content}'''"
        response = Generation.call(
            model=self.dashscope_model,
            messages=[{'role': 'user', 'content': sum_prompt}],
            result_format='message',
        )
        
        if response.status_code == 200:
            return response.output.choices[0]['message']['content']
        else:
            raise Exception(f"DashScope API error: {response.code} - {response.message}")

    def get_help_text(self, verbose, **kwargs):
        apis = ", ".join(self.available_apis)
        current = self.preferred_api
        return f'使用jina reader和AI模型（当前使用: {current}, 可用: {apis}）总结网页链接内容'

    def _load_config_template(self):
        logger.debug("No Suno plugin config.json, use plugins/jina_sum/config.json.template")
        try:
            plugin_config_path = os.path.join(self.path, "config.json.template")
            if os.path.exists(plugin_config_path):
                with open(plugin_config_path, "r", encoding="utf-8") as f:
                    plugin_conf = json.load(f)
                    return plugin_conf
        except Exception as e:
            logger.exception(e)

    def _get_jina_url(self, target_url):
        return self.jina_reader_base + "/" + target_url

    def _get_openai_chat_url(self):
        return self.open_ai_api_base + "/chat/completions"

    def _get_openai_headers(self):
        return {
            'Authorization': f"Bearer {self.open_ai_api_key}",
            'Host': urlparse(self.open_ai_api_base).netloc
        }

    def _get_openai_payload(self, content: str):
        sum_prompt = f"{self.prompt}\n\n'''{content}'''"
        messages = [{"role": "user", "content": sum_prompt}]
        payload = {
            'model': self.open_ai_model,
            'messages': messages
        }
        return payload

    def _check_url(self, target_url: str):
        stripped_url = target_url.strip()
        # 简单校验是否是url
        if not stripped_url.startswith("http://") and not stripped_url.startswith("https://"):
            return False

        # 检查白名单
        if len(self.white_url_list):
            if not any(stripped_url.startswith(white_url) for white_url in self.white_url_list):
                return False

        # 排除黑名单，黑名单优先级>白名单
        for black_url in self.black_url_list:
            if stripped_url.startswith(black_url):
                return False

        return True
