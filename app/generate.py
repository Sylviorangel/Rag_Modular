from typing import Iterable, Optional
from openai import OpenAI

class OpenAIChat:
    def __init__(self, api_key: str, api_base: str | None, model: str):
        kwargs = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.model = model

    def complete(self, messages: list[dict], temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict], temperature: float = 0.2) -> Iterable[str]:
        with self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        ) as stream:
            for event in stream:
                if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
