"""
Pydantic schemas for request / response bodies.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. NVDA")
    date: str | None = Field(
        None,
        description="Analysis date YYYY-MM-DD. Defaults to today if omitted.",
    )
    llm_provider: str | None = Field(
        None,
        description="LLM provider to use for this request. Defaults to ollama. Supported: ollama, google, openrouter.",
    )
    deep_model: str | None = Field(
        None,
        description="Override the deep-think model id for this request only. Falls back to the provider default when omitted.",
    )
    quick_model: str | None = Field(
        None,
        description="Override the quick-think model id for this request only. Falls back to the provider default when omitted.",
    )


class SubmitResponse(BaseModel):
    request_id: str
    ticker: str
    analysis_date: str
    llm_provider: str
    status: str
    submitted_at: str


class RequestStatus(BaseModel):
    request_id: str
    ticker: str
    analysis_date: str
    llm_provider: str | None = None
    deep_model: str | None = None
    quick_model: str | None = None
    status: str
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    recommendation: str | None = None
    llm_calls: int | None = None
    tool_calls: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    agent_recommendations: dict | None = None
    analysis_url: str | None = None
    debug_log_url: str | None = None
    error_message: str | None = None


class RequestListResponse(BaseModel):
    total: int
    requests: list[RequestStatus]


class CancelResponse(BaseModel):
    request_id: str
    status: str
    canceled_at: str


class CancelAllResponse(BaseModel):
    canceled_count: int
    canceled_at: str


class BatchScheduleCreateRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. NVDA")
    llm_provider: str = Field(..., description="Provider for scheduled runs: ollama, google, or openrouter")
    frequency: str = Field(..., description="Run frequency: daily, weekly, or monthly")
    deep_model: str | None = Field(None, description="Override deep-think model id (ollama only)")
    quick_model: str | None = Field(None, description="Override quick-think model id (ollama only)")


class BatchScheduleRerunRequest(BaseModel):
    llm_provider: str = Field(..., description="Provider to use for this rerun: ollama, google, or openrouter")
    deep_model: str | None = Field(None, description="Override deep-think model id for this rerun (ollama only)")
    quick_model: str | None = Field(None, description="Override quick-think model id for this rerun (ollama only)")


class BatchScheduleUpdateRequest(BaseModel):
    llm_provider: str = Field(..., description="Updated provider for future scheduled runs")
    frequency: str = Field(..., description="Updated frequency for future scheduled runs: daily, weekly, or monthly")
    deep_model: str | None = Field(None, description="Updated deep-think model id override (ollama only)")
    quick_model: str | None = Field(None, description="Updated quick-think model id override (ollama only)")


class BatchScheduleItem(BaseModel):
    id: str
    ticker: str
    llm_provider: str
    frequency: str
    deep_model: str | None = None
    quick_model: str | None = None
    next_run_at: str | None = None
    last_schedule_run_at: str | None = None
    latest_recommendation: str | None = None
    last_run_at: str | None = None
    latest_logs_url: str | None = None
    latest_analysis_url: str | None = None


class BatchScheduleListResponse(BaseModel):
    total: int
    schedules: list[BatchScheduleItem]


class EnvVarUpdateRequest(BaseModel):
    value: str = Field(..., description="New env variable value")


class EnvVarValueResponse(BaseModel):
    name: str
    value: str | None = None
    exists: bool


class OllamaEndpoint(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    url: str = Field(..., min_length=1, max_length=255)


class RuntimeSettingsUpdateRequest(BaseModel):
    max_workers: int = Field(1, ge=1, le=32)
    ollama_endpoints: list[OllamaEndpoint] = Field(..., min_length=1, max_length=32)


class VaultRefreshResponse(BaseModel):
    enabled: bool
    updated: int
    keys: list[str]
    skipped: list[str]
    message: str


class LatestRecommendationResponse(BaseModel):
    ticker: str
    provider: str | None = None
    available: bool
    latest: dict | None = None
