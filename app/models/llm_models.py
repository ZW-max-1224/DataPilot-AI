from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    system_prompt: str = Field(default="You are a helpful assistant.", min_length=1, max_length=12_000)
    user_prompt: str = Field(min_length=1, max_length=20_000)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=8_192)


class LLMUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    model: str = Field(min_length=1)
    usage: LLMUsage | None = None
    finish_reason: str | None = None
    mode: Literal["llm", "demo"]
