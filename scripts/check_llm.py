"""Check that the agent-side LLM configuration can call DeepSeek."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from autoresearch_agent.context import Context
from autoresearch_agent.utils import load_chat_model


async def main() -> None:
    """Invoke the configured chat model once."""
    load_dotenv()
    context = Context()
    model = load_chat_model(
        context.model,
        base_url=context.model_base_url,
        api_key_env=context.model_api_key_env,
    )
    response = await model.ainvoke(
        "Reply with exactly this text: deepseek-ok",
    )
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
