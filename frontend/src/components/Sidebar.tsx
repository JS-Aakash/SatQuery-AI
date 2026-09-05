"use client";

import React from "react";
import {
  Satellite,
  Compass,
  Layers,
  History,
  FileText,
  BarChart3,
  Settings,
  Cpu,
  Activity,
  ShieldCheck,
  User,
  ExternalLink,
} from "lucide-react";
import { cn } from "../lib/utils";

export type NavTab = "home" | "workspace" | "history" | "reports" | "evaluation" | "settings";

interface SidebarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  onOpenSystemStatus: () => void;
  backendOnline: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  onOpenSystemStatus,
  backendOnline,
}) => {
  const navItems = [
    { id: "home" as NavTab, label: "New Analysis", icon: Compass, badge: "Core" },
    { id: "workspace" as NavTab, label: "Workspace", icon: Layers },
    { id: "history" as NavTab, label: "Analysis History", icon: History },
    { id: "reports" as NavTab, label: "Saved Reports", icon: FileText },
    { id: "evaluation" as NavTab, label: "Evaluation", icon: BarChart3, badge: "Benchmarks" },
    { id: "settings" as NavTab, label: "Settings", icon: Settings },
  ];

  return (
    <aside className="w-64 bg-[#080d18] border-r border-slate-800/80 flex flex-col justify-between shrink-0 select-none z-30 h-screen sticky top-0">
      {/* Brand Header */}
      <div>
        <div className="p-5 border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onTabChange("home")}>
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-500/20 via-cyan-500/20 to-blue-600/30 border border-emerald-500/40 flex items-center justify-center shadow-lg shadow-emerald-950/40">
              <Satellite className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <div className="font-semibold text-sm tracking-wider text-slate-100 uppercase flex items-center gap-1.5">
                <span>SatQuery AI</span>
                <span className="text-[10px] px-1 py-0.2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded font-mono">v1.0</span>
              </div>
              <p className="text-[11px] text-slate-400 font-mono tracking-tight">ISRO MISSION INTELLIGENCE</p>
            </div>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1">
          <div className="px-3 py-1.5 text-[10px] font-mono tracking-wider text-slate-500 uppercase">
            Platform Views
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={cn(
                  "w-full flex items-center justify-between px-3 py-2 rounded-md text-xs font-medium transition-all duration-150 group",
                  isActive
                    ? "bg-slate-800/90 text-emerald-400 border border-emerald-500/30 shadow-sm"
                    : "text-slate-300 hover:text-slate-100 hover:bg-slate-800/50"
                )}
              >
                <div className="flex items-center space-x-3">
                  <Icon
                    className={cn(
                      "w-4 h-4 transition-colors",
                      isActive ? "text-emerald-400" : "text-slate-400 group-hover:text-slate-200"
                    )}
                  />
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={cn(
                      "text-[10px] px-1.5 py-0.5 rounded font-mono",
                      isActive
                        ? "bg-emerald-500/20 text-emerald-300"
                        : "bg-slate-800 text-slate-400"
                    )}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Mission Control Telemetry */}
      <div className="p-3 border-t border-slate-800/80 space-y-2 bg-[#060a13]">
        {/* Model Status Card */}
        <div className="px-3 py-2 rounded-md bg-slate-900/80 border border-slate-800 text-[11px] space-y-1 font-mono">
          <div className="flex items-center justify-between text-slate-400">
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              Specialist Models
            </span>
            <span className="text-amber-400 text-[10px] bg-amber-500/10 px-1 py-0.5 rounded border border-amber-500/20">
              Adapters Pending
            </span>
          </div>
          <p className="text-[10px] text-slate-500 leading-tight">
            BigEarthNet / RSVQA training ready for integration.
          </p>
        </div>

        {/* System Status Action Button */}
        <button
          onClick={onOpenSystemStatus}
          className="w-full px-3 py-2 rounded-md bg-slate-900/60 hover:bg-slate-800 border border-slate-800 flex items-center justify-between text-xs transition-colors group"
        >
          <div className="flex items-center space-x-2">
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                backendOnline ? "bg-emerald-400 radar-live-dot" : "bg-rose-500"
              )}
            />
            <span className="font-mono text-[11px] text-slate-300">System Status</span>
          </div>
          <span
            className={cn(
              "text-[10px] font-mono px-1.5 py-0.5 rounded",
              backendOnline
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                : "bg-rose-500/10 text-rose-400 border border-rose-500/30"
            )}
          >
            {backendOnline ? "ONLINE" : "OFFLINE"}
          </span>
        </button>

        {/* User / Profile Section */}
        <div className="pt-2 px-1 flex items-center justify-between text-xs">
          <div className="flex items-center space-x-2.5">
            <div className="w-7 h-7 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300">
              <User className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-slate-200 font-medium text-xs leading-none">ISRO Commander</div>
              <div className="text-[10px] text-slate-500 font-mono mt-0.5">SAC / Geo-Ops</div>
            </div>
          </div>
          <ShieldCheck className="w-4 h-4 text-emerald-500/80" />
        </div>
      </div>
    </aside>
  );
};
