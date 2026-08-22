import os
from typing import List, Union, Optional, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import AsyncOpenAI

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.utils.env import load_env

load_env()

def _msg_content(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content", ""))
    return str(getattr(item, "content", "") or "")

def _messages_to_openai(msg: List[Any]) -> List[dict]:
    out: List[dict] = []
    for m in msg:
        if isinstance(m, Message):
            out.append({"role": m.role, "content": m.content})
        elif isinstance(m, dict):
            out.append(
                {
                    "role": m.get("role", "user"),
                    "content": str(m.get("content", "")),
                }
            )
        else:
            out.append({"role": "user", "content": str(m)})
    return out

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Any],
    max_tokens: int = 1000,
    temperature: float = 0.6,
) -> str:
    api_key = os.getenv("API_KEY") or ""
    base_url = os.getenv("BASE_URL") or None
    if base_url == "":
        base_url = None

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    messages = _messages_to_openai(msg)

    request_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    disable_thinking = os.getenv("GDESIGNER_DISABLE_THINKING", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if disable_thinking:
        model_l = model.lower()
        base_l = (base_url or "").lower()

        if ("aliyuncs.com" in base_l or "dashscope" in base_l) and "qwen" in model_l:
            request_kwargs["extra_body"] = {"enable_thinking": False}

        elif "qwen3.5" in model_l:


            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        elif "qwen3.6" in model_l:
            request_kwargs["extra_body"] = {"enable_thinking": False}

    response = await client.chat.completions.create(
        **request_kwargs,
    )
    completion = (response.choices[0].message.content or "").strip()
    prompt = "".join(_msg_content(item) for item in msg)
    cost_count(prompt, completion, model)
    return completion

@LLMRegistry.register("GPTChat")
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(
            self.model_name,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
