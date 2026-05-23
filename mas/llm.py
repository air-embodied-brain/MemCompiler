import os

from typing import (
    Protocol,
    Literal,
    Optional,
    List,
    Any,
)
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .utils import load_config
from .azure import azure_openai_model


# model configs
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

# Only read OpenAI env vars if they exist (for Azure compatibility)
URL = os.environ.get("OPENAI_BASE_URL", "")
KEY = os.environ.get("OPENAI_API_KEY", "")
if URL or KEY:
    print('# api url: ', URL)
    print('# api key: ', KEY)


completion_tokens, prompt_tokens = 0, 0


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> Optional[int]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    return int(value)


def _build_gemini_thinking_config(genai_types: Any, model_name: str) -> Optional[Any]:
    if not _env_flag("MEMCOMPILER_GEMINI_DISABLE_THINKING", False):
        return None

    thinking_level = os.environ.get("MEMCOMPILER_GEMINI_THINKING_LEVEL", "").strip()
    thinking_budget = _env_optional_int("MEMCOMPILER_GEMINI_THINKING_BUDGET")
    thinking_kwargs: dict[str, Any] = {}

    if thinking_budget is not None:
        thinking_kwargs["thinking_budget"] = thinking_budget
    elif thinking_level:
        thinking_kwargs["thinking_level"] = thinking_level
    else:
        model_name_lower = model_name.lower()
        if "gemini-3" in model_name_lower:
            # Gemini 3 Flash does not expose a full "off" switch; use the lowest safe level.
            thinking_kwargs["thinking_level"] = "minimal"
        elif "gemini-2.5-flash" in model_name_lower:
            thinking_kwargs["thinking_budget"] = 0
        else:
            print(
                f"# Gemini thinking disable requested for {model_name}, "
                "but no safe default is configured for this model. Using provider default."
            )
            return None

    print(f"# Gemini thinking config for {model_name}: {thinking_kwargs}")
    return genai_types.ThinkingConfig(**thinking_kwargs)

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class GPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.client = OpenAI(
            base_url=URL,
            api_key=KEY
        )

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        import time
        global prompt_tokens, completion_tokens

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Ensure max_tokens is never None
        if max_tokens is None:
            max_tokens = MAX_TOKEN

        max_retries = 5
        wait_time = 1

        for attempt in range(max_retries):
            try:
                # Disable thinking mode for Qwen3/3.5 models to get direct action output
                extra_body = {}
                if "qwen3" in self.model_name.lower():
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}

                # GPT-5 uses max_completion_tokens instead of max_tokens and doesn't support stop parameter
                if "gpt-5" in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        # temperature=temperature,
                        n=num_comps
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=num_comps,
                        stop=stop_strs
                    )

                # print(f"DEBUG: response type: {type(response)}")
                # print(f"DEBUG: response content: {response}")
                answer = response.choices[0].message.content
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

                if answer is None:
                    print("Error: LLM returned None")
                    continue
                return answer

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during API call: {error_message}")
                    break

        return ""


class LocalLLM(LLM):
    """Local LLM via vLLM OpenAI-compatible API"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1"):
        super().__init__(model_name=model_name)
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"  # vLLM doesn't require a real key
        )

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        import time
        global prompt_tokens, completion_tokens

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Ensure max_tokens is never None
        if max_tokens is None:
            max_tokens = MAX_TOKEN

        max_retries = 5
        wait_time = 1

        for attempt in range(max_retries):
            # try:
            #     # GPT-5 uses max_completion_tokens instead of max_tokens and doesn't support stop parameter
            #     if "gpt-5" in self.model_name.lower():
            #         response = self.client.chat.completions.create(
            #             model=self.model_name,
            #             messages=messages,
            #             max_completion_tokens=max_tokens,
            #             # temperature=temperature,
            #             n=num_comps
            #         )
            #     else:
            #         response = self.client.chat.completions.create(
            #             model=self.model_name,
            #             messages=messages,
            #             max_tokens=max_tokens,
            #             temperature=temperature,
            #             n=num_comps,
            #             stop=stop_strs
            #         )

            #     answer = response.choices[0].message.content
            #     prompt_tokens += response.usage.prompt_tokens
            #     completion_tokens += response.usage.completion_tokens

            #     if answer is None:
            #         print("Error: LLM returned None")
            #         continue
            #     return answer

            # except Exception as e:
            #     error_message = str(e)
            #     if "rate limit" in error_message.lower() or "429" in error_message:
            #         time.sleep(wait_time)
            #     else:
            #         print(f"Error during API call: {error_message}")
            #         break

            # Disable thinking mode for Qwen3/3.5 models to get direct action output
            extra_body = {}
            if "qwen3" in self.model_name.lower():
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}

            if "gpt-5" in self.model_name.lower():
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    # temperature=temperature,
                    n=num_comps
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_comps,
                    stop=stop_strs,
                    extra_body=extra_body if extra_body else None
                )

            answer = response.choices[0].message.content
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens

            if answer is None:
                print("Error: LLM returned None")
                continue
            return answer


        return ""


class AzureGPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.client = azure_openai_model(model_name)

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        import time
        global prompt_tokens, completion_tokens

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Ensure max_tokens is never None
        if max_tokens is None:
            max_tokens = MAX_TOKEN

        max_retries = 5
        wait_time = 1

        for attempt in range(max_retries):
            try:
                # GPT-5 uses max_completion_tokens instead of max_tokens and doesn't support stop parameter
                if "gpt-5" in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        # temperature=temperature,
                        n=num_comps
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=num_comps,
                        stop=stop_strs
                    )

                answer = response.choices[0].message.content
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

                if answer is None:
                    print("Error: LLM returned None")
                    continue
                return answer

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during API call: {error_message}")
                    break

        return ""


class GeminiChat(LLM):
    """Gemini LLM via google-genai SDK. Reads GEMINI_API_KEY and GOOGLE_GEMINI_BASE_URL from env."""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        from google import genai
        from google.genai import types as genai_types
        self._genai_types = genai_types

        api_key = os.environ.get("GEMINI_API_KEY", "")
        base_url = os.environ.get("GOOGLE_GEMINI_BASE_URL", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["http_options"] = {"base_url": base_url}
            print(f"# Gemini base_url: {base_url}")
        self.client = genai.Client(**client_kwargs)
        print(f"# Gemini model: {model_name}")

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        import time
        global prompt_tokens, completion_tokens

        # Separate system instruction from conversation
        system_parts = []
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "user":
                contents.append(self._genai_types.Content(
                    role="user", parts=[self._genai_types.Part(text=msg.content)]
                ))
            elif msg.role == "assistant":
                contents.append(self._genai_types.Content(
                    role="model", parts=[self._genai_types.Part(text=msg.content)]
                ))

        if max_tokens is None:
            max_tokens = MAX_TOKEN

        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        thinking_config = _build_gemini_thinking_config(
            genai_types=self._genai_types,
            model_name=self.model_name,
        )
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        if system_parts:
            config_kwargs["system_instruction"] = "\n".join(system_parts)
        if stop_strs:
            config_kwargs["stop_sequences"] = stop_strs

        config = self._genai_types.GenerateContentConfig(**config_kwargs)

        max_retries = 5
        wait_time = 1

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                answer = response.text
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    prompt_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    completion_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

                if answer is None:
                    print("Error: Gemini returned None")
                    continue
                return answer

            except Exception as e:
                error_message = str(e)
                if "429" in error_message or "rate" in error_message.lower():
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    print(f"Error during Gemini API call: {error_message}")
                    break

        return ""


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000
