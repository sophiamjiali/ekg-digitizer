# ==============================================================================
# File Name : schemas.py
# Purpose   : Request/response models for the API
# ==============================================================================

from pydantic import BaseModel, Field


class LeadSignal(BaseModel):
    n_samples: int = Field(description="Number of samples in this lead")
    duration_s: float = Field(description="Lead duration in seconds")
    values: list[float] = Field(description="Signal in millivolts")


class DigitizeResponse(BaseModel):
    model_version: str = Field(description="Registry version of the model that produced this result")
    fs: int = Field(description="Sampling frequency in Hz of every lead")
    units: str = "mV"
    leads: dict[str, LeadSignal]


class Health(BaseModel):
    status: str = "ok"


class Ready(BaseModel):
    status: str
    model_version: str | None = None
    device: str | None = None


class ErrorResponse(BaseModel):
    detail: str