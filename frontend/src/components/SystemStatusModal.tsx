"use client";

import React, { useState, useEffect } from "react";
import {
  Activity,
  X,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Cpu,
  Server,
  Terminal,
  Shield,
  RefreshCw,
  ExternalLink,
  Zap,
  Trash2,
  Loader2,
} from "lucide-react";
import { SystemStatusResponse, SubsystemStatus, ModelHardwareStatus } from "../lib/types";
import { fetchSystemStatus, fetchModelHardwareStatus, preloadModel, unloadModel } from "../lib/api";
import { cn } from "../lib/utils";

interface SystemStatusModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SystemStatusModal: React.FC<SystemStatusModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [statusData, setStatusData] = useState<SystemStatusResponse | null>(null);
  const [modelHw, setModelHw] = useState<ModelHardwareStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  const loadStatus = async () => {
    setLoading(true);
    try {
      const [data, hw] = await Promise.all([
        fetchSystemStatus(),
        fetchModelHardwareStatus().catch(() => null),
      ]);
      setStatusData(data);
      setModelHw(hw);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handlePreload = async () => {
    setActionLoading(true);
    try {
      const res = await preloadModel();
      setModelHw(res);
    } catch (e) {
      console.error(e);
    } finally {
      setActionLoading(false);
    }
  };

  const handleUnload = async () => {
    setActionLoading(true);
    try {
      const res = await unloadModel();
      setModelHw(res);
    } catch (e) {
      console.error(e);
    } finally {
      setActionLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadStatus();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "READY":
        return {
          label: "READY",
          className: "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30",
          icon: CheckCircle2,
        };
      case "NOT CONFIGURED":
        return {
          label: "NOT CONFIGURED",
          className: "bg-amber-500/10 text-amber-400 border border-amber-500/30",
          icon: Clock,
        };
      case "OFFLINE":
        return {
          label: "OFFLINE",
          className: "bg-rose-500/10 text-rose-400 border border-rose-500/30",
          icon: AlertTriangle,
        };
      default:
        return {
          label: status,
          className: "bg-slate-800 text-slate-400 border border-slate-700",
          icon: AlertTriangle,
        };
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-150">
      <div className="relative w-full max-w-3xl bg-[#080d18] border border-slate-700/80 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="p-4 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center space-x-2.5">
            <Activity className="w-5 h-5 text-emerald-400" />
            <div>
              <h3 className="text-sm font-semibold text-slate-100 uppercase tracking-wider font-mono">
                SatQuery AI System Diagnostics & Telemetry
              </h3>
              <p className="text-[11px] text-slate-500 font-mono">
                Truthful status of platform subsystems & specialist model readiness
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={loadStatus}
              className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
              title="Refresh Telemetry"
            >
              <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="p-5 overflow-y-auto space-y-4">
          {/* AI Model Hardware & GPU Residency Card */}
          {modelHw && (
            <div className="p-4 rounded-lg bg-gradient-to-r from-slate-900/90 to-slate-950 border border-slate-700/80 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Cpu className="w-4 h-4 text-emerald-400" />
                  <span className="text-xs font-mono font-bold text-slate-200 tracking-wider">
                    GPU ACCELERATOR & AI MODEL RESIDENCY
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <span
                    className={cn(
                      "text-[10px] font-mono px-2 py-0.5 rounded font-bold border",
                      modelHw.is_warm
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/40"
                        : "bg-amber-500/10 text-amber-400 border-amber-500/40"
                    )}
                  >
                    {modelHw.is_warm ? "🟢 WARM IN VRAM (FAST ~2s)" : "🟡 STANDBY (COLD ~28s)"}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px] font-mono">
                <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-500 text-[10px]">DEVICE</div>
                  <div className="text-slate-300 font-semibold truncate" title={modelHw.device_name}>
                    {modelHw.device_name.split(" ")[0]} {modelHw.device_name.split(" ").slice(1, 4).join(" ")}
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-500 text-[10px]">VRAM RESIDENCY</div>
                  <div className="text-emerald-400 font-semibold">
                    {modelHw.vram_allocated_gb} / {modelHw.vram_gb} GB
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-500 text-[10px]">QUANTIZATION</div>
                  <div className="text-cyan-400 font-semibold">
                    {modelHw.recommended_precision}
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-500 text-[10px]">PRIMARY MODEL</div>
                  <div className="text-slate-300 font-semibold truncate">
                    GeoChat-7B (4-bit)
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between pt-1 border-t border-slate-800/60">
                <span className="text-[10px] text-slate-400 font-mono">
                  {modelHw.is_warm
                    ? "Model is warm in VRAM. Subsequent analyses execute in ~2.1 seconds."
                    : "Preload model to keep weights warm in GPU and avoid 30s cold load."}
                </span>
                <div className="flex items-center space-x-2">
                  {modelHw.is_warm ? (
                    <button
                      onClick={handleUnload}
                      disabled={actionLoading}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] font-mono text-slate-300 flex items-center space-x-1.5 transition-colors disabled:opacity-50"
                    >
                      {actionLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3 text-rose-400" />}
                      <span>Unload VRAM</span>
                    </button>
                  ) : (
                    <button
                      onClick={handlePreload}
                      disabled={actionLoading}
                      className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-[10px] font-mono text-white font-semibold flex items-center space-x-1.5 transition-colors shadow-sm disabled:opacity-50"
                    >
                      {actionLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3 fill-white" />}
                      <span>Warmup GPU (Preload)</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Subsystem Status Table */}
          <div className="grid grid-cols-1 gap-2.5">
            {statusData?.subsystems.map((sub) => {
              const badge = getStatusBadge(sub.status);
              const Icon = badge.icon;
              return (
                <div
                  key={sub.name}
                  className="p-3.5 rounded-lg bg-slate-900/70 border border-slate-800 flex items-start justify-between gap-4"
                >
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-semibold text-slate-200 font-mono">
                        {sub.name}
                      </span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-950 text-slate-400 border border-slate-800">
                        {sub.component_id}
                      </span>
                      {sub.version && (
                        <span className="text-[10px] font-mono text-slate-500">
                          v{sub.version}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400">
                      {sub.description}
                    </p>
                    {sub.notes && (
                      <p className="text-[10px] text-amber-400 font-mono">
                        ⓘ {sub.notes}
                      </p>
                    )}
                  </div>

                  <div className="text-right shrink-0">
                    <span
                      className={cn(
                        "text-[10px] font-mono font-bold px-2 py-1 rounded flex items-center space-x-1",
                        badge.className
                      )}
                    >
                      <Icon className="w-3 h-3" />
                      <span>{badge.label}</span>
                    </span>
                    {sub.latency_ms !== null && sub.latency_ms !== undefined && (
                      <div className="text-[10px] text-slate-500 font-mono mt-1">
                        Ping: {sub.latency_ms}ms
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="p-3 rounded bg-slate-900/40 border border-slate-800/80 text-[11px] text-slate-400 font-mono space-y-1">
            <div className="text-slate-300 font-semibold uppercase tracking-wider">
              Integration Architecture Policy:
            </div>
            <p>
              In accordance with Module 1 specifications, specialist model components
              (GeoChat VQA, CDVQA Temporal, Optical-SAR Fusion) are marked accurately as{" "}
              <span className="text-amber-400 font-bold">NOT CONFIGURED</span> until
              their fine-tuned checkpoints are registered in Modules 3–5.
            </p>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-3 bg-slate-900/80 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-200 transition-colors"
          >
            Close Diagnostics
          </button>
        </div>
      </div>
    </div>
  );
};
