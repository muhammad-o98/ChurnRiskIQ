"""Pydantic schemas and input validators."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, validator


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: int = Field(ge=0, le=1)
    Dependents: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0)
    PhoneService: int = Field(ge=0, le=1)
    MultipleLines: int = Field(ge=0, le=1)
    InternetService: str
    OnlineSecurity: int = Field(ge=0, le=1)
    OnlineBackup: int = Field(ge=0, le=1)
    DeviceProtection: Optional[int] = 0
    TechSupport: int = Field(ge=0, le=1)
    StreamingTV: int = Field(ge=0, le=1)
    StreamingMovies: int = Field(ge=0, le=1)
    Contract: str
    PaperlessBilling: int = Field(ge=0, le=1)
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    @validator('gender')
    def gender_valid(cls, v):
        if v not in {'Male', 'Female'}:
            raise ValueError('gender must be Male or Female')
        return v

    @validator('InternetService')
    def svc_valid(cls, v):
        if v not in {'DSL', 'Fiber optic', 'No'}:
            raise ValueError('InternetService must be DSL, Fiber optic, or No')
        return v

    @validator('Contract')
    def contract_valid(cls, v):
        if v not in {'Month-to-month', 'One year', 'Two year'}:
            raise ValueError('Contract invalid')
        return v
