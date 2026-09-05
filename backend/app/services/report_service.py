"""
Report Generation Service for SatQuery AI.
Exports structured analysis reports in Markdown, JSON, and audit formats.
"""
import json
import uuid
from datetime import datetime, timezone

from .base import ReportService
from ..schemas import ReportGenerationRequest, ReportResponse, AnalysisResponse


class DefaultReportService(ReportService):
    """
    Generates auditable analysis reports.
    """

    def generate_report(self, request: ReportGenerationRequest, analysis: AnalysisResponse) -> ReportResponse:
        report_id = f"rep_{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()
        title = f"SatQuery AI Mission Report - {analysis.task.value}"

        if request.format.lower() == "json":
            content_dict = {
                "report_id": report_id,
                "created_at": created_at,
                "analysis_id": analysis.analysis_id,
                "query": analysis.query,
                "task": analysis.task.value,
                "status": analysis.status,
                "calibrated_confidence": analysis.confidence,
                "answer": analysis.answer,
                "evidence": [e.model_dump() for e in analysis.evidence] if request.include_evidence else [],
                "execution_trace": [t.model_dump() for t in analysis.execution_trace] if request.include_trace else [],
                "models_used": analysis.models_used,
                "execution_time_ms": analysis.execution_time_ms,
                "analyst_notes": request.analyst_notes or "Automated agentic inference report."
            }
            content = json.dumps(content_dict, indent=2)
            download_filename = f"SatQuery_Report_{analysis.analysis_id}.json"
        else:
            # Markdown Report
            lines = [
                f"# {title}",
                f"**Generated**: {created_at} | **Analysis ID**: `{analysis.analysis_id}`",
                "",
                "---",
                "",
                "## 1. Executive Summary",
                f"- **Natural-Language Query**: *\"{analysis.query}\"*",
                f"- **Identified Task**: `{analysis.task.value}`",
                f"- **Calibrated Confidence**: **{analysis.confidence_formatted}**",
                f"- **Execution Latency**: `{analysis.execution_time_ms} ms`",
                f"- **Models Employed**: {', '.join(analysis.models_used)}",
                "",
                "## 2. Findings & Natural Language Answer",
                f"> {analysis.answer}",
                ""
            ]

            if request.include_evidence and analysis.evidence:
                lines.extend([
                    "## 3. Grounded Visual & Spectral Evidence",
                    "| ID | Category | Feature Title | Confidence | Description |",
                    "| :--- | :--- | :--- | :--- | :--- |"
                ])
                for ev in analysis.evidence:
                    lines.append(f"| `{ev.id}` | {ev.category} | **{ev.title}** | {int(ev.confidence*100)}% | {ev.description} |")
                lines.append("")

            if request.include_trace and analysis.execution_trace:
                lines.extend([
                    "## 4. Auditable Agent Execution Trace",
                    "| Step | Action | Specialist / Tool | Duration | Parameters & Output Summary |",
                    "| :--- | :--- | :--- | :--- | :--- |"
                ])
                for st in analysis.execution_trace:
                    lines.append(f"| {st.step_id} | {st.name} | `{st.tool_or_model or 'System'}` | {st.duration_ms}ms | {st.output_summary or '-'} |")
                lines.append("")

            if request.analyst_notes:
                lines.extend([
                    "## 5. Analyst Annotations",
                    f"{request.analyst_notes}",
                    ""
                ])

            lines.extend([
                "---",
                "*SatQuery AI Remote Sensing Platform — Built for ISRO / Department of Space problem statement 26167.*"
            ])
            content = "\n".join(lines)
            download_filename = f"SatQuery_Report_{analysis.analysis_id}.md"

        return ReportResponse(
            report_id=report_id,
            analysis_id=analysis.analysis_id,
            format=request.format,
            title=title,
            created_at=created_at,
            content=content,
            download_filename=download_filename
        )
