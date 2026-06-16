"""Agent-side utility functions."""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def get_message_text(msg: BaseMessage) -> str:
    """Return the text content of a LangChain message."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return "".join(c if isinstance(c, str) else str(c.get("text") or "") for c in content)


def load_chat_model(
    fully_specified_name: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> BaseChatModel:
    """Load an agent-side chat model from ``provider/model`` syntax."""
    provider, model = fully_specified_name.split("/", maxsplit=1)
    if provider == "deepseek":
        api_key = os.environ.get(api_key_env or "DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing DeepSeek API key. Set DEEPSEEK_API_KEY in .env or your shell."
            )
        return ChatOpenAI(
            model=model,
            base_url=base_url or "https://api.deepseek.com",
            api_key=SecretStr(api_key),
            temperature=0,
        )
    return init_chat_model(model, model_provider=provider)
