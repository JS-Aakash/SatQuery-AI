"use client";

import React, { useState, useEffect } from "react";
import {
  ChevronDown,
  Layers,
  Activity,
  Globe,
  Bell,
  Moon,
  Shield,
  HelpCircle,
  Zap,
  Loader2,
} from "lucide-react";
import { cn } from "../lib/utils";
import { fetchModelHardwareStatus, preloadModel } from "../lib/api";
import { ModelHardwareStatus } from "../lib/types";

interface TopNavProps {
  currentWorkspace: string;
  onSelectWorkspace: (ws: string) => void;
  onOpenSystemStatus: () => void;
  backendOnline: boolean;
  latencyMs?: number;
}

export const TopNav: React.FC<TopNavProps> = ({
  currentWorkspace,
  onSelectWorkspace,
  onOpenSystemStatus,
  backendOnline,
  latencyMs = 28,
}) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [modelHw, setModelHw] = useState<ModelHardwareStatus | null>(null);
  const [isWarmingUp, setIsWarmingUp] = useState(false);

  useEffect(() => {
    if (!backendOnline) return;
    const checkHw = () => {
      fetchModelHardwareStatus()
        .then(setModelHw)
        .catch(() => {});
    };
    checkHw();
    const interval = setInterval(checkHw, 8000);
    return () => clearInterval(interval);
  }, [backendOnline]);

  const handleWarmup = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsWarmingUp(true);
    try {
      const res = await preloadModel();
      setModelHw(res);
    } catch (err) {
      console.error("Warmup failed:", err);
    } finally {
      setIsWarmingUp(false);
    }
  };

  const workspaces = [
    "ISRO SAC Cartosat/RISAT Evaluation Set",
    "Sentinel-1/2 Coastal & Harbor AOI",
    "Bangalore Urban Sprawl Bi-Temporal",
    "Brahmaputra Flood Hydrological Monitoring",
  ];

  return (
    <header className="h-14 bg-[#080d18]/90 backdrop-blur-md border-b border-slate-800/80 px-6 flex items-center justify-between sticky top-0 z-20">
      {/* Left: Workspace Selector */}
      <div className="relative flex items-center space-x-3">
        <div className="flex items-center space-x-2 text-xs text-slate-400 font-mono">
          <Globe className="w-3.5 h-3.5 text-cyan-400" />
          <span>WORKSPACE:</span>
        </div>

        <button
          onClick={() => setDropdownOpen(!dropdownOpen)}
          className="flex items-center space-x-2 px-3 py-1.5 rounded-md bg-slate-900/90 hover:bg-slate-800 border border-slate-700/70 text-xs text-slate-200 font-medium transition-all shadow-sm"
        >
          <span className="truncate max-w-[280px]">{currentWorkspace}</span>
          <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
        </button>

        {dropdownOpen && (
          <div className="absolute top-11 left-24 w-80 bg-slate-900 border border-slate-700 rounded-md shadow-2xl p-1 z-50 animate-in fade-in zoom-in-95 duration-100">
            <div className="px-3 py-1.5 text-[10px] font-mono text-slate-400 uppercase border-b border-slate-800">
              Select Mission Workspace
            </div>
            {workspaces.map((ws) => (
              <button
                key={ws}
                onClick={() => {
                  onSelectWorkspace(ws);
                  setDropdownOpen(false);
                }}
                className={cn(
                  "w-full text-left px-3 py-2 text-xs rounded hover:bg-slate-800 transition-colors flex items-center space-x-2",
                  ws === currentWorkspace
                    ? "text-emerald-400 font-medium bg-slate-800/40"
                    : "text-slate-300"
                )}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                <span className="truncate">{ws}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Right: Telemetry & Actions */}
      <div className="flex items-center space-x-3">
        {/* GPU Warmth & Residency Pill */}
        {modelHw && (
          modelHw.is_warm ? (
            <div
              title={`GeoChat-7B is warm & resident in VRAM (${modelHw.vram_allocated_gb} GB allocated on ${modelHw.device_name}). Fast ~2s inference ready.`}
              className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-500/40 text-[11px] font-mono text-emerald-300"
            >
              <Zap className="w-3 h-3 text-emerald-400 fill-emerald-400" />
              <span>GPU WARM</span>
              <span className="text-emerald-500/80">({modelHw.vram_allocated_gb}GB)</span>
            </div>
          ) : (
            <button
              onClick={handleWarmup}
              disabled={isWarmingUp}
              title="Preload GeoChat-7B weights into GPU VRAM for instant ~2s inference"
              className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-cyan-950/40 hover:bg-cyan-900/50 border border-cyan-500/50 text-[11px] font-mono text-cyan-300 transition-colors shadow-sm cursor-pointer disabled:opacity-60"
            >
              {isWarmingUp ? (
                <>
                  <Loader2 className="w-3 h-3 text-cyan-400 animate-spin" />
                  <span>WARMING GPU...</span>
                </>
              ) : (
                <>
                  <Zap className="w-3 h-3 text-cyan-400" />
                  <span>WARMUP GPU (~2s fast)</span>
                </>
              )}
            </button>
          )
        )}

        {/* Backend Latency Pill */}
        <button
          onClick={onOpenSystemStatus}
          className="flex items-center space-x-2 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-[11px] font-mono transition-colors hover:border-slate-700"
        >
          <span
            className={cn(
              "w-2 h-2 rounded-full",
              backendOnline ? "bg-emerald-400 radar-live-dot" : "bg-rose-500"
            )}
          />
          <span className="text-slate-400">BACKEND:</span>
          <span className={backendOnline ? "text-emerald-400" : "text-rose-400"}>
            {backendOnline ? `${latencyMs}ms` : "OFFLINE"}
          </span>
        </button>

        {/* System Status Pill */}
        <button
          onClick={onOpenSystemStatus}
          className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-800/70 hover:bg-slate-800 border border-slate-700 text-xs text-slate-300 font-mono transition-colors"
        >
          <Activity className="w-3.5 h-3.5 text-cyan-400" />
          <span>SYSTEM DIAGNOSTICS</span>
        </button>

        {/* Theme Status (Fixed Dark-First) */}
        <div
          title="Satellite Mission Control is configured in Dark-First High Contrast mode"
          className="p-1.5 rounded bg-slate-900 border border-slate-800 text-slate-400 cursor-default"
        >
          <Moon className="w-3.5 h-3.5 text-slate-300" />
        </div>
      </div>
    </header>
  );
};
