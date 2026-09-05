"""
Schemas for analysis history, reporting, and evaluation metrics.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .analysis import TaskType


class AnalysisHistoryItem(BaseModel):
    id: str
    timestamp: str
    query: str
    location: str
    imagery_type: str
    task: TaskType
    models_used: List[str]
    confidence: float
    confidence_formatted: str
    status: str = "COMPLETED"
    preview_url: Optional[str] = None
    has_change_map: bool = False
    grounding_count: int = 0


class AnalysisHistoryResponse(BaseModel):
    total: int
    items: List[AnalysisHistoryItem]


class ReportGenerationRequest(BaseModel):
    analysis_id: str
    format: str = Field(default="markdown", description="'markdown', 'json', or 'pdf'")
    include_evidence: bool = True
    include_trace: bool = True
    analyst_notes: Optional[str] = None


class ReportResponse(BaseModel):
    report_id: str
    analysis_id: str
    format: str
    title: str
    created_at: str
    content: str
    download_filename: str


class BenchmarkScore(BaseModel):
    metric: str
    value: float
    target: float
    unit: str = "%"
    dataset: str
    description: str


class EvaluationMetricsResponse(BaseModel):
    overall_accuracy: float = 88.4
    vqa_performance: BenchmarkScore
    grounding_performance: BenchmarkScore
    change_detection_performance: BenchmarkScore
    optical_sar_performance: BenchmarkScore
    average_latency_ms: int = 1420
    confidence_calibration_ece: float = 0.042
    historical_benchmark_runs: List[Dict[str, Any]]
    is_mock_evaluation: bool = True
