"use client";

import React, { useState } from "react";
import {
  Send,
  Sparkles,
  Command,
  CornerDownLeft,
  Loader2,
  Terminal,
} from "lucide-react";
import { cn } from "../lib/utils";

interface QueryBoxProps {
  query: string;
  onQueryChange: (q: string) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  disabled?: boolean;
}

export const QueryBox: React.FC<QueryBoxProps> = ({
  query,
  onQueryChange,
  onAnalyze,
  isAnalyzing,
  disabled = false,
}) => {
  const suggestions = [
    "What land-cover types are visible?",
    "Highlight the water bodies.",
    "What changed between these two dates?",
    "Has built-up area increased?",
    "Compare optical and SAR information.",
  ];

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !isAnalyzing && !disabled) {
        onAnalyze();
      }
    }
  };

  return (
    <div className="space-y-3">
      {/* Query Input Container */}
      <div className="relative rounded-xl bg-slate-900/90 border border-slate-700/80 shadow-2xl focus-within:border-emerald-500/80 focus-within:ring-1 focus-within:ring-emerald-500/30 transition-all p-3.5">
        <div className="flex items-start space-x-3">
          <Terminal className="w-5 h-5 text-emerald-400 mt-1 shrink-0" />
          <div className="flex-1">
            <textarea
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isAnalyzing || disabled}
              rows={2}
              placeholder="Ask something about your satellite imagery (e.g. land-cover classification, region grounding, temporal changes, optical-SAR fusion)..."
              className="w-full bg-transparent text-slate-100 text-sm placeholder:text-slate-500 resize-none focus:outline-none leading-relaxed font-sans"
            />
          </div>
        </div>

        {/* Footer actions inside box */}
        <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 mt-1">
          <div className="flex items-center space-x-1.5 text-[11px] text-slate-500 font-mono">
            <Command className="w-3 h-3" />
            <span>Press</span>
            <kbd className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 text-[10px]">
              Enter ↵
            </kbd>
            <span>to analyze</span>
          </div>

          <button
            onClick={onAnalyze}
            disabled={!query.trim() || isAnalyzing || disabled}
            className={cn(
              "px-5 py-2 rounded-lg font-semibold text-xs flex items-center space-x-2 transition-all shadow-md",
              query.trim() && !isAnalyzing && !disabled
                ? "bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-emerald-950/50 hover:shadow-emerald-900/60 cursor-pointer"
                : "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50"
            )}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>ORCHESTRATING AGENT...</span>
              </>
            ) : (
              <>
                <span>ANALYZE</span>
                <CornerDownLeft className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Suggested Queries Chips */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-mono text-slate-500 flex items-center gap-1">
          <Sparkles className="w-3 h-3 text-amber-400" />
          Suggestions:
        </span>
        {suggestions.map((sug) => (
          <button
            key={sug}
            onClick={() => onQueryChange(sug)}
            className="px-2.5 py-1 rounded-full text-xs font-mono bg-slate-900/80 hover:bg-slate-800 text-slate-300 hover:text-emerald-400 border border-slate-800 hover:border-slate-700 transition-colors"
          >
            {sug}
          </button>
        ))}
      </div>
    </div>
  );
};
