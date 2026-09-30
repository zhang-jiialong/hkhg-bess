import base64
import copy
import json
import os
import re
import struct
import asyncio
from collections.abc import Callable as AbcCallable
from functools import lru_cache
from typing import List, Dict, Callable, Any, Union, Optional
import aioboto3
import aiohttp
import numpy as np
import ollama
import torch
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,

    RetryCallState,
)

from tenacity.wait import wait_base


from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)

import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _coerce_pathlike_model_name(model_name: str) -> str:
    return os.path.abspath(model_name) if model_name and os.path.exists(model_name) else model_name


class CustomWait(wait_base):
    def __call__(self, retry_state: RetryCallState):
        exception = retry_state.outcome.exception()
        if isinstance(exception, RateLimitError):

            return min(4 * (2 ** max(retry_state.attempt_number - 1, 0)), 60)
        else:

            return min(1 * (2 ** max(retry_state.attempt_number - 1, 0)), 10)


def summarize_openai_error(exc: Exception) -> str:
    if exc is None:
        return "unknown error"

    parts: list[str] = [exc.__class__.__name__]
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(f"status={status_code}")

    response = getattr(exc, "response", None)
    if response is not None and status_code is None:
        response_status = getattr(response, "status_code", None)
        if response_status is not None:
            parts.append(f"status={response_status}")

    message = getattr(exc, "message", None)
    if message:
        parts.append(f"message={message}")
    elif str(exc):
        parts.append(f"message={str(exc)}")

    body = getattr(exc, "body", None)
    if body:
        try:
            body_text = json.dumps(body, ensure_ascii=False)
        except TypeError:
            body_text = str(body)
        if len(body_text) > 500:
            body_text = body_text[:500] + "..."
        parts.append(f"body={body_text}")
    elif response is not None:
        try:
            body_text = response.text
        except Exception:
            body_text = None
        if body_text:
            if len(body_text) > 500:
                body_text = body_text[:500] + "..."
            parts.append(f"body={body_text}")

    return " | ".join(parts)


def _log_openai_retry_before_sleep(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    next_sleep = getattr(getattr(retry_state, "next_action", None), "sleep", None)
    if next_sleep is None:
        next_sleep = 0.0
    logger.warning(
        "Retryable OpenAI error on attempt %s, sleeping %.2fs. %s",
        retry_state.attempt_number,
        float(next_sleep),
        summarize_openai_error(exc),
    )


def openai_retry(
    *,
    attempts: int = 5,
    wait_strategy: Optional[wait_base] = None,
) -> AbcCallable[[AbcCallable[..., Any]], AbcCallable[..., Any]]:
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_strategy or CustomWait(),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
        before_sleep=_log_openai_retry_before_sleep,
        reraise=True,
    )


@openai_retry()
async def openai_complete_if_cache(
    model,
    prompt,
    count_token=False,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url is not None:
        client_kwargs["base_url"] = base_url

    openai_async_client = AsyncOpenAI(**client_kwargs)
    close_client_here = True
    try:
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        logger.debug("===== Query Input to LLM =====")
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug("Full context:")
        if "response_format" in kwargs:
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )

        logger.debug("===== Response from LLM =====")
        token_usage = dict()
        if hasattr(response, "usage"):
            token_usage = {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', None),
                "completion_tokens": getattr(response.usage, 'completion_tokens', None),
                "total_tokens": getattr(response.usage, 'total_tokens', None),
            }
            logger.debug(f"Token usage: {token_usage}")

        if hasattr(response, "__aiter__"):
            async def inner():
                try:
                    async for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content is None:
                            continue
                        if r"\u" in content:
                            content = safe_unicode_decode(content.encode("utf-8"))
                        logger.debug(f"Response: {content}")
                        yield content
                finally:
                    await openai_async_client.close()

            close_client_here = False
            if count_token:
                return inner(), token_usage
            return inner()

        content = response.choices[0].message.content
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        logger.debug(f"Response: {content}")
        if count_token:
            return content, token_usage
        return content
    finally:
        if close_client_here:
            await openai_async_client.close()


async def openai_compatible_complete(
    model_name,
    prompt,
    count_token=False,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    **kwargs,
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        model_name,
        prompt,
        count_token=count_token,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@openai_retry()
async def dash_openai_complete_if_cache(
    model,
    prompt,
    count_token=False,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
) -> str:

    model = "qwen3-8b-instruct" if not model else model

    DASHSCOPE_BASE_URL_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY missing")


    openai_async_client = AsyncOpenAI(api_key=key, base_url=DASHSCOPE_BASE_URL_INTL)


    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})

    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Model: {model}")
    logger.debug(f"Base URL: {base_url}")
    logger.debug(f"System prompt: {system_prompt}")


    if "response_format" in kwargs:
        kwargs.pop("response_format", None)
        logger.warning("Dropped response_format for DashScope (unsupported).")

    kwargs.setdefault("extra_body", {}).setdefault("enable_thinking", False)

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )


    token_usage = {}
    if hasattr(response, "usage"):
        token_usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }
        logger.debug(f"Token usage: {token_usage}")

    if hasattr(response, "__aiter__"):
        async def inner():
            async for chunk in response:
                delta = getattr(chunk.choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if not content:
                    continue
                yield content
        return (inner(), token_usage) if count_token else inner()

    content = response.choices[0].message.content
    return (content, token_usage) if count_token else content


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]


async def openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt, count_token=False, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        count_token=count_token,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, count_token=False, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        count_token=count_token,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_35_turbo(
    prompt, count_token=False, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-3.5-turbo-0125",
        prompt,
        count_token=count_token,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4_turbo(
    prompt, count_token=False, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4-turbo-2024-04-09",
        prompt,
        count_token=count_token,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def qwen3_8b(
    prompt, count_token=False, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await dash_openai_complete_if_cache(
        "qwen3-8b",
        prompt,
        count_token=count_token,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@openai_retry()
async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    timeout: float | None = None,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    client_kwargs["max_retries"] = 0

    openai_async_client = AsyncOpenAI(**client_kwargs)
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer

    resolved_model_name = _coerce_pathlike_model_name(model_name)
    kwargs = {"trust_remote_code": True}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(resolved_model_name, **kwargs)


def make_sentence_transformer_embedding_func(
    model_name: str,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    batch_size: int = 8,
):
    resolved_model_name = _coerce_pathlike_model_name(model_name)
    model = _load_sentence_transformer(resolved_model_name, device)
    embedding_dim = model.get_sentence_embedding_dimension()
    tokenizer = getattr(model, "tokenizer", None)
    max_token_size = getattr(tokenizer, "model_max_length", 8192) if tokenizer else 8192

    @wrap_embedding_func_with_attrs(
        embedding_dim=int(embedding_dim),
        max_token_size=int(max_token_size if max_token_size and max_token_size > 0 else 8192),
    )
    async def _sentence_transformer_embedding(texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        def _encode():
            return model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )

        embeddings = await asyncio.to_thread(_encode)
        return np.asarray(embeddings, dtype=np.float32)

    return _sentence_transformer_embedding


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete("How are you?")
        print(result)

    asyncio.run(main())
