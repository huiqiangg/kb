import json
from collections.abc import AsyncIterator

import httpx

from app.core.config import Settings


class ModelClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.settings.model_access_key:
            headers["accessKey"] = self.settings.model_access_key
        return headers

    @staticmethod
    def _content_from_payload(data: str) -> str:
        """兼容 OpenAI SSE delta 与平台返回的非流式 choices.message。"""
        payload = json.loads(data)
        choice = payload.get("choices", [{}])[0]
        return choice.get("delta", {}).get("content") or choice.get("message", {}).get("content", "")

    async def stream_rewrite(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """消费改写模型的 SSE，调用方可把内部进度即时反馈给前端。"""
        async with self.client.stream(
            "POST", self.settings.rewrite_model_url, headers=self._headers(),
            json={"messages": messages, "model": self.settings.rewrite_model_name, "stream": True},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = line.removeprefix("data:").strip() if line.startswith("data:") else line
                if data == "[DONE]":
                    return
                try:
                    text = self._content_from_payload(data)
                    if text:
                        yield text
                except (json.JSONDecodeError, IndexError, AttributeError):
                    continue

    async def extract_tags(self, messages: list[dict[str, str]]) -> list[str]:
        response = await self.client.post(
            self.settings.rewrite_model_url,
            headers=self._headers(),
            json={"messages": messages, "model": self.settings.rewrite_model_name, "stream": False},
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        try:
            value = json.loads(text[text.find("{") : text.rfind("}") + 1])
            return [str(tag) for tag in value.get("tags", []) if str(tag).strip()][:8]
        except (json.JSONDecodeError, AttributeError, KeyError):
            return []

    async def stream_answer(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if not self.settings.answer_model_url:
            raise RuntimeError("未配置 ANSWER_MODEL_URL")
        async with self.client.stream(
            "POST", self.settings.answer_model_url, headers=self._headers(),
            json={"messages": messages, "model": self.settings.answer_model_name, "stream": True},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = line.removeprefix("data:").strip() if line.startswith("data:") else line
                if data == "[DONE]":
                    return
                try:
                    text = self._content_from_payload(data)
                    if text:
                        yield text
                except (json.JSONDecodeError, IndexError, AttributeError):
                    continue
