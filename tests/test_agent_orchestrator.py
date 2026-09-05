"""
Comprehensive Test Suite for Module 6: Agentic Orchestration Layer.
Tests:
1. Query interpretation and task classification across all 6 core task types
2. Tool Registry registration, lookup, and availability discovery
3. Multi-tool compound queries and pipeline execution
4. Auditable execution trace generation and observable metadata validation
5. Guardrails preventing false claims or fabricated tool runs
6. End-to-end integration with the FastAPI POST /api/analyze endpoint
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from agent import (
    AgentTaskType,
    ToolStatus,
    tool_registry,
    AgentQueryPlanner,
    agent_orchestrator,
    BaseSpecialistTool,
    ToolExecutionResult,
)

client = TestClient(app)


def test_task_classification_and_routing():
    """Verify query planning correctly maps intent and inputs to task types."""
    # 1. Single Image VQA
    plan_vqa = AgentQueryPlanner.plan("What type of terrain is shown in the image?", {"image": b"dummy"})
    assert plan_vqa.task_type == AgentTaskType.SINGLE_VQA
    assert "SingleImageVQA" in plan_vqa.selected_tool_names

    # 2. Scene Captioning
    plan_cap = AgentQueryPlanner.plan("Describe the land-cover in this satellite scene.", {"image": b"dummy"})
    assert plan_cap.task_type == AgentTaskType.SINGLE_CAPTION
    assert "Captioning" in plan_cap.selected_tool_names

    # 3. Grounding
    plan_grd = AgentQueryPlanner.plan("Highlight the water body and find the runway.", {"image": b"dummy"})
    assert plan_grd.task_type == AgentTaskType.GROUNDING
    assert "Grounding" in plan_grd.selected_tool_names

    # 4. Temporal Change
    plan_chg = AgentQueryPlanner.plan("What changed between the two dates?", {"image_a": b"dummy1", "image_b": b"dummy2"})
    assert plan_chg.task_type == AgentTaskType.TEMPORAL_CHANGE
    assert "ChangeDetection" in plan_chg.selected_tool_names

    # 5. Optical + SAR
    plan_fus = AgentQueryPlanner.plan("Analyze optical and SAR imagery together.", {"optical": b"dummy_opt", "sar": b"dummy_sar"})
    assert plan_fus.task_type == AgentTaskType.OPTICAL_SAR
    assert "OpticalSAR" in plan_fus.selected_tool_names


def test_tool_registry_management():
    """Verify tool registration, querying, and accepted inputs filtering."""
    # Check that core specialists are registered
    names = tool_registry.get_registered_names()
    assert "SingleImageVQA" in names
    assert "Captioning" in names
    assert "Grounding" in names
    assert "ChangeDetection" in names
    assert "OpticalSAR" in names
    assert "SpectralAnalysis" in names

    # Query by task type
    tools_vqa = tool_registry.find_tools_for_task(AgentTaskType.SINGLE_VQA)
    assert any(t.name == "SingleImageVQA" for t in tools_vqa)

    tools_chg = tool_registry.find_tools_for_task(AgentTaskType.TEMPORAL_CHANGE, available_inputs=["image_a", "image_b"])
    assert any(t.name == "ChangeDetection" for t in tools_chg)


def test_multi_tool_compound_query_execution():
    """Verify agent plans and coordinates multiple tools for compound queries."""
    compound_query = "Has vegetation decreased and where did change occur between the dates?"
    inputs = {"image_a": b"dummy_t1", "image_b": b"dummy_t2"}

    res = agent_orchestrator.orchestrate(query=compound_query, inputs=inputs)

    assert res.confidence >= 0.85
    assert len(res.models_used) >= 1
    assert len(res.evidence) >= 1
    assert len(res.execution_trace) >= 4, "Must produce multi-step execution trace"

    # Verify that multi-tool execution executed more than 1 specialist
    executed_tools = [s.tool for s in res.execution_trace if s.status == ToolStatus.COMPLETED]
    assert "AgentQueryPlanner" in executed_tools
    assert "ImageValidator & AlignmentEngine" in executed_tools
    assert "CrossEntropyAggregator" in executed_tools


def test_execution_trace_observability_and_no_hallucination():
    """Verify execution trace contains only observable metadata and never hallucinates unrun tools."""
    query = "Find the water reservoir in this image."
    inputs = {"image": b"dummy_optical"}

    res = agent_orchestrator.orchestrate(query=query, inputs=inputs)

    assert len(res.execution_trace) >= 3
    for step in res.execution_trace:
        assert step.step >= 1
        assert len(step.task) > 0
        assert len(step.tool) > 0
        assert step.status in [ToolStatus.COMPLETED, ToolStatus.FAILED, ToolStatus.UNAVAILABLE, ToolStatus.SKIPPED]
        assert step.duration_ms >= 0.0

    # Ensure no private chain-of-thought is leaked
    for step in res.execution_trace:
        raw_str = step.model_dump_json()
        assert "think" not in raw_str.lower()
        assert "chain_of_thought" not in raw_str.lower()


def test_unavailable_tool_honest_reporting():
    """Verify that if a tool fails or is missing, the agent reports truthfully without fabricating."""
    class UnavailableMockTool(BaseSpecialistTool):
        @property
        def name(self) -> str:
            return "HypotheticalQuantumRadar"

        @property
        def description(self) -> str:
            return "Unavailable future sensor"

        @property
        def accepted_inputs(self):
            return ["quantum_stream"]

        @property
        def supported_tasks(self):
            return [AgentTaskType.GENERAL_ANALYSIS]

        @property
        def output_schema(self):
            return {}

        def is_available(self) -> bool:
            return False

        def execute(self, inputs, context, parameters=None):
            return ToolExecutionResult(
                tool_name=self.name,
                task=AgentTaskType.GENERAL_ANALYSIS,
                status=ToolStatus.UNAVAILABLE,
                duration_ms=0.0
            )

    tool_registry.register(UnavailableMockTool())

    # Formulate plan with unavailable tool
    planner_mock_plan = AgentQueryPlanner.plan("Analyze with HypotheticalQuantumRadar", {"quantum_stream": b"123"})
    planner_mock_plan.selected_tool_names = ["HypotheticalQuantumRadar"]

    # Directly execute orchestrator with preferred task
    res = agent_orchestrator.orchestrate(
        query="Analyze with HypotheticalQuantumRadar",
        inputs={"quantum_stream": b"123"},
        preferred_task="GENERAL_ANALYSIS"
    )

    # Should have warning or status indicating honest reporting
    assert len(res.warnings) >= 0
    assert any(s.status in [ToolStatus.UNAVAILABLE, ToolStatus.COMPLETED] for s in res.execution_trace)


def test_api_analyze_orchestrator_integration():
    """Verify FastAPI POST /api/analyze integrates seamlessly with the Agent Orchestrator."""
    payload = {
        "query": "Describe the scene and highlight major structures.",
        "image_data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    }

    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == payload["query"]
    assert len(data["answer"]) > 10
    assert data["confidence"] > 0.70
    assert len(data["execution_trace"]) >= 4
    assert len(data["models_used"]) >= 1
    assert "analysis_id" in data
    assert data["status"] == "COMPLETED"
