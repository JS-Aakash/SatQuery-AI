"use client";

import React, { useState, useEffect } from "react";
import {
  History,
  Search,
  Trash2,
  FileText,
  ExternalLink,
  CheckCircle2,
  Calendar,
  Layers,
  Sparkles,
  RefreshCw,
  Cpu,
} from "lucide-react";
import { AnalysisHistoryItem, TaskType } from "../lib/types";
import { fetchAnalyses, deleteAnalysis } from "../lib/api";
import { cn } from "../lib/utils";

interface HistoryViewProps {
  onOpenAnalysis: (id: string) => void;
  onOpenReportModal: (id: string) => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  onOpenAnalysis,
  onOpenReportModal,
}) => {
  const [items, setItems] = useState<AnalysisHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterTask, setFilterTask] = useState<string>("ALL");
  const [searchFilter, setSearchFilter] = useState("");

  const loadHistory = async () => {
    setLoading(true);
    const data = await fetchAnalyses();
    setItems(data);
    setLoading(false);
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm("Are you sure you want to remove this analysis record from session history?")) {
      const ok = await deleteAnalysis(id);
      if (ok) {
        setItems((prev) => prev.filter((item) => item.id !== id));
      }
    }
  };

  const filteredItems = items.filter((item) => {
    const matchesTask = filterTask === "ALL" || item.task === filterTask;
    const matchesSearch =
      item.query.toLowerCase().includes(searchFilter.toLowerCase()) ||
      item.location.toLowerCase().includes(searchFilter.toLowerCase());
    return matchesTask && matchesSearch;
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-5 rounded-xl bg-slate-900/60 border border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <History className="w-5 h-5 text-emerald-400" />
            <h2 className="text-sm font-semibold text-slate-100 uppercase tracking-wider font-mono">
              Mission Analysis History & Audit Trail
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Review past agentic remote-sensing queries, confidence scores, and specialist execution logs.
          </p>
        </div>

        <button
          onClick={loadHistory}
          className="px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono flex items-center space-x-1.5 transition-colors border border-slate-700"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2 bg-slate-950 border border-slate-800 rounded-md px-3 py-1.5 w-72">
          <Search className="w-3.5 h-3.5 text-slate-500" />
          <input
            type="text"
            placeholder="Search query, location..."
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            className="w-full bg-transparent text-xs text-slate-200 focus:outline-none"
          />
        </div>

        {/* Task Filter Pills */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs font-mono">
          {["ALL", "Visual Question Answering", "Text-Guided Region Grounding", "Bi-Temporal Change Analysis", "Cross-Modal Optical-SAR Analysis"].map((t) => (
            <button
              key={t}
              onClick={() => setFilterTask(t)}
              className={cn(
                "px-2.5 py-1 rounded transition-colors",
                filterTask === t
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                  : "bg-slate-900/60 text-slate-400 hover:text-slate-200 border border-slate-800"
              )}
            >
              {t === "ALL" ? "All Tasks" : t.replace(" Analysis", "")}
            </button>
          ))}
        </div>
      </div>

      {/* History Grid */}
      {loading ? (
        <div className="py-20 text-center space-y-3 font-mono text-xs text-slate-500">
          <RefreshCw className="w-6 h-6 animate-spin mx-auto text-emerald-400" />
          <p>LOADING HISTORICAL MISSIONS...</p>
        </div>
      ) : filteredItems.length === 0 ? (
        <div className="p-12 text-center rounded-xl bg-slate-900/40 border border-dashed border-slate-800 space-y-2">
          <History className="w-8 h-8 text-slate-600 mx-auto" />
          <p className="text-xs text-slate-400">No historical analyses match current filter.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3">
          {filteredItems.map((item) => (
            <div
              key={item.id}
              onClick={() => onOpenAnalysis(item.id)}
              className="p-4 rounded-xl bg-slate-900/70 hover:bg-slate-900 border border-slate-800/90 hover:border-slate-700 transition-all cursor-pointer flex flex-col md:flex-row md:items-center justify-between gap-4 group shadow-sm"
            >
              <div className="space-y-1.5 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                    {item.task}
                  </span>
                  <span className="text-[10px] font-mono text-slate-500 flex items-center gap-1">
                    <Calendar className="w-3 h-3 text-cyan-400" />
                    {item.timestamp.split("T")[0]}
                  </span>
                  <span className="text-[10px] font-mono text-slate-500">
                    📍 {item.location}
                  </span>
                </div>

                <h3 className="text-sm font-semibold text-slate-200 group-hover:text-emerald-400 transition-colors">
                  "{item.query}"
                </h3>

                <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono text-slate-400">
                  <span className="text-slate-500">Models:</span>
                  <span>{item.models_used.join(", ")}</span>
                  {item.has_change_map && (
                    <span className="text-amber-400 bg-amber-500/10 px-1.5 py-0.2 rounded border border-amber-500/20">
                      Change Map Available
                    </span>
                  )}
                  {item.grounding_count > 0 && (
                    <span className="text-cyan-400 bg-cyan-500/10 px-1.5 py-0.2 rounded border border-cyan-500/20">
                      {item.grounding_count} Grounded Objects
                    </span>
                  )}
                </div>
              </div>

              {/* Right: Metrics & Actions */}
              <div className="flex items-center space-x-4 shrink-0">
                <div className="text-right">
                  <div className="text-[10px] font-mono text-slate-500">CONFIDENCE</div>
                  <div className="text-emerald-400 font-mono font-bold text-sm">
                    {item.confidence_formatted}
                  </div>
                </div>

                <div className="flex items-center space-x-1 border-l border-slate-800 pl-3">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onOpenReportModal(item.id);
                    }}
                    title="Export Mission Report"
                    className="p-2 rounded hover:bg-slate-800 text-slate-400 hover:text-emerald-400 transition-colors"
                  >
                    <FileText className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => handleDelete(item.id, e)}
                    title="Delete record"
                    className="p-2 rounded hover:bg-slate-800 text-slate-400 hover:text-rose-400 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
