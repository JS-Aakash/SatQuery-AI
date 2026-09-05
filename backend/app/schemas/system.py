"""
Pydantic schemas for subsystem status and diagnostic telemetry.
"""
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class StatusEnum(str, Enum):
    READY = "READY"
    OFFLINE = "OFFLINE"
    NOT_CONFIGURED = "NOT CONFIGURED"
    ERROR = "ERROR"


class SubsystemStatus(BaseModel):
    name: str
    component_id: str
    status: StatusEnum
    version: Optional[str] = None
    latency_ms: Optional[int] = None
    description: str
    notes: Optional[str] = None


class SystemStatusResponse(BaseModel):
    overall_status: StatusEnum
    timestamp: str
    subsystems: List[SubsystemStatus]
    environment: str = "Development / Pre-Model Integration"
    active_workspace: str = "Default Geospatial Workspace"
