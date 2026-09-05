"use client";

import React, { useState, useEffect } from "react";
import {
  FileText,
  X,
  Download,
  Copy,
  Check,
  RefreshCw,
  FileCode,
  FileSpreadsheet,
} from "lucide-react";
import { generateReport } from "../lib/api";
import { ReportResponse } from "../lib/types";
import { cn } from "../lib/utils";

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  analysisId: string | null;
}

export const ReportModal: React.FC<ReportModalProps> = ({
  isOpen,
  onClose,
  analysisId,
}) => {
  const [format, setFormat] = useState<"markdown" | "json">("markdown");
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (isOpen && analysisId) {
      setLoading(true);
      generateReport(analysisId, format)
        .then((data) => setReport(data))
        .finally(() => setLoading(false));
    }
  }, [isOpen, analysisId, format]);

  if (!isOpen || !analysisId) return null;

  const handleCopy = () => {
    if (report?.content) {
      navigator.clipboard.writeText(report.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownload = () => {
    if (!report) return;
    const blob = new Blob([report.content], {
      type: format === "json" ? "application/json" : "text/markdown;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = report.download_filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-150">
      <div className="relative w-full max-w-3xl bg-[#080d18] border border-slate-700/80 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="p-4 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center space-x-2.5">
            <FileText className="w-5 h-5 text-emerald-400" />
            <div>
              <h3 className="text-sm font-semibold text-slate-100 uppercase tracking-wider font-mono">
                SatQuery AI Mission Intelligence Report
              </h3>
              <p className="text-[11px] text-slate-500 font-mono">
                Export auditable evidence trace, telemetry metrics, and geospatial findings
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Format Selector Bar */}
        <div className="px-5 py-2.5 bg-slate-950/60 border-b border-slate-800/80 flex items-center justify-between text-xs font-mono">
          <div className="flex items-center space-x-2">
            <span className="text-slate-500 uppercase">DOCUMENT FORMAT:</span>
            <button
              onClick={() => setFormat("markdown")}
              className={cn(
                "px-2.5 py-1 rounded transition-colors flex items-center space-x-1",
                format === "markdown"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Markdown (.md)</span>
            </button>
            <button
              onClick={() => setFormat("json")}
              className={cn(
                "px-2.5 py-1 rounded transition-colors flex items-center space-x-1",
                format === "json"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <FileCode className="w-3.5 h-3.5" />
              <span>JSON Schema (.json)</span>
            </button>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleCopy}
              disabled={!report || loading}
              className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors flex items-center space-x-1"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied" : "Copy"}</span>
            </button>
            <button
              onClick={handleDownload}
              disabled={!report || loading}
              className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-semibold transition-colors flex items-center space-x-1 shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download</span>
            </button>
          </div>
        </div>

        {/* Report Content Preview */}
        <div className="p-5 overflow-y-auto flex-1 font-mono text-xs text-slate-300 bg-slate-950/80">
          {loading ? (
            <div className="py-20 text-center space-y-2 text-slate-500">
              <RefreshCw className="w-6 h-6 animate-spin mx-auto text-emerald-400" />
              <p>COMPILING AUDITABLE MISSION REPORT...</p>
            </div>
          ) : report ? (
            <pre className="whitespace-pre-wrap leading-relaxed select-text">
              {report.content}
            </pre>
          ) : (
            <div className="py-12 text-center text-slate-500">
              No report data available.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
