"use client";

import React, { useState } from "react";
import {
  Search,
  Calendar,
  CloudSun,
  Satellite,
  Check,
  ArrowRight,
  Filter,
  CheckCircle2,
  RefreshCw,
  Sparkles,
  Zap,
  Layers,
  MapPin,
  Clock,
  FileCheck,
} from "lucide-react";
import { searchSatelliteScenes, executeAutoLocationQuery } from "../lib/api";
import { ImageMetadata } from "../lib/types";
import { PRESET_SCENARIOS } from "../lib/presets";
import { cn } from "../lib/utils";

interface SatelliteSearchProps {
  onSelectScene: (meta: ImageMetadata, isSecondary?: boolean) => void;
  onExecuteAutoAnalysis?: (result: any) => void;
}

export const SatelliteSearch: React.FC<SatelliteSearchProps> = ({
  onSelectScene,
  onExecuteAutoAnalysis,
}) => {
  // Query states
  const [autoQueryText, setAutoQueryText] = useState("Compare Chennai between January 2024 and January 2026");
  const [isAutoExecuting, setIsAutoExecuting] = useState(false);
  const [autoResponse, setAutoResponse] = useState<any | null>(null);

  // Manual search parameters
  const [location, setLocation] = useState("Chennai, Tamil Nadu");
  const [startDate, setStartDate] = useState("2024-01-01");
  const [endDate, setEndDate] = useState("2026-01-01");
  const [selectedSensors, setSelectedSensors] = useState<string[]>(["Sentinel-2", "Sentinel-1"]);
  const [cloudCover, setCloudCover] = useState(15);
  const [imageryType, setImageryType] = useState("Level-2A Bottom-of-Atmosphere (BOA) Reflectance");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<any[] | null>(null);
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [selectedSecondarySceneId, setSelectedSecondarySceneId] = useState<string | null>(null);

  const toggleSensor = (sensor: string) => {
    setSelectedSensors((prev) =>
      prev.includes(sensor) ? prev.filter((s) => s !== sensor) : [...prev, sensor]
    );
  };

  // 1-Click Autonomous Retrieval & Analysis
  const handleRunAutoQuery = async () => {
    if (!autoQueryText.trim()) return;
    setIsAutoExecuting(true);
    setAutoResponse(null);
    try {
      const res = await executeAutoLocationQuery({
        query: autoQueryText.trim(),
        max_cloud_coverage: cloudCover,
      });
      setAutoResponse(res);
      if (onExecuteAutoAnalysis) {
        onExecuteAutoAnalysis(res);
      }
    } catch (err: any) {
      console.error("Auto query failed:", err);
    } finally {
      setIsAutoExecuting(false);
    }
  };

  // Search Catalog for Scenes
  const handleSearch = async () => {
    setIsSearching(true);
    setSelectedSceneId(null);
    setSelectedSecondarySceneId(null);
    try {
      const res = await searchSatelliteScenes({
        start_date: startDate,
        end_date: endDate,
        sensors: selectedSensors,
        cloud_coverage: cloudCover,
      });
      setSearchResults(res.scenes || []);
    } catch (err) {
      // Fallback
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleApplyScene = (scene: any, isSecondary: boolean = false) => {
    if (isSecondary) {
      setSelectedSecondarySceneId(scene.id || scene.scene_id);
    } else {
      setSelectedSceneId(scene.id || scene.scene_id);
    }

    const isSar = (scene.sensor || "").toLowerCase().includes("sar") || (scene.sensor || "").toLowerCase().includes("s1");
    const meta: ImageMetadata = {
      id: scene.id || scene.scene_id,
      filename: `${scene.id || "Sentinel_Scene"}.tif`,
      file_size_bytes: scene.file_size_bytes || 480 * 1024 * 1024,
      file_size_formatted: scene.file_size_formatted || "480 MB",
      format: "GeoTIFF",
      dimensions: [1024, 1024],
      bands: isSar ? 2 : 12,
      band_names: isSar ? ["VV", "VH"] : ["B02 Blue", "B03 Green", "B04 Red", "B08 NIR", "B11 SWIR"],
      crs: scene.metadata?.crs || "EPSG:32644 (WGS 84 / UTM Zone 44N)",
      bounding_box: scene.bbox || [80.18, 13.00, 80.35, 13.25],
      modality: isSar ? "SAR" : "Optical",
      sensor: isSar ? "Sentinel-1 SAR C-Band" : "Sentinel-2 MSI",
      acquisition_date: scene.acquisition_date || new Date().toISOString(),
      preview_url: scene.preview_url,
      is_valid: true,
      validation_notes: [scene.selection_reason || "Copernicus CDSE Catalog Authenticated Scene"],
    };

    onSelectScene(meta, isSecondary);
  };

  return (
    <div className="space-y-6">
      {/* 1. Autonomous Earth Observation Query Card */}
      <div className="p-5 rounded-xl bg-gradient-to-r from-emerald-950/40 via-slate-900/80 to-cyan-950/40 border border-emerald-500/30 space-y-4 shadow-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <h3 className="text-xs font-semibold text-emerald-300 uppercase tracking-wider font-mono">
              Module 7: Autonomous Earth Observation Retrieval
            </h3>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            Copernicus CDSE / Sentinel Hub Live
          </span>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed">
          Ask for any Earth location and time window. The autonomous pipeline identifies the AOI coordinates, queries Copernicus, selects optimal cloud-free scenes, downloads and caches the rasters, and executes multimodal intelligence.
        </p>

        <div className="flex flex-col sm:flex-row gap-2">
          <div className="relative flex-1">
            <input
              type="text"
              value={autoQueryText}
              onChange={(e) => setAutoQueryText(e.target.value)}
              placeholder="e.g. Compare Chennai between January 2024 and January 2026"
              className="w-full bg-slate-950 border border-slate-700/80 rounded-lg px-3.5 py-2.5 text-slate-100 text-xs focus:outline-none focus:border-emerald-400 transition-colors font-mono"
            />
          </div>

          <button
            onClick={handleRunAutoQuery}
            disabled={isAutoExecuting || !autoQueryText.trim()}
            className="px-5 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-xs font-mono flex items-center justify-center space-x-2 transition-all shadow-lg shadow-emerald-950/50 disabled:opacity-50 shrink-0"
          >
            {isAutoExecuting ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>RETRIEVING & ANALYZING...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>EXECUTE AUTO PIPELINE</span>
              </>
            )}
          </button>
        </div>

        {/* Quick Suggestion Pills */}
        <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px] font-mono text-slate-400">
          <span className="text-slate-500">QUICK SAMPLES:</span>
          {[
            "Compare Chennai between January 2024 and January 2026",
            "Analyze Kochi backwaters SAR imagery",
            "Inspect Mumbai harbor Sentinel-2 optical scene",
          ].map((s) => (
            <button
              key={s}
              onClick={() => setAutoQueryText(s)}
              className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-[10px] border border-slate-700/80 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>

        {/* Auto Execution Result Snippet */}
        {autoResponse && (
          <div className="p-4 rounded-lg bg-slate-950/90 border border-emerald-500/40 text-xs space-y-3 font-mono animate-in fade-in duration-200">
            <div className="flex items-center justify-between text-emerald-400 font-semibold">
              <span className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" /> Autonomous Execution Successful ({autoResponse.location_name})
              </span>
              <span className="text-slate-400 text-[10px]">
                {autoResponse.execution_time_ms} ms
              </span>
            </div>

            <p className="text-slate-200 font-sans text-xs leading-relaxed">
              {autoResponse.answer}
            </p>

            {/* Execution Trace Steps */}
            <div className="space-y-1.5 pt-2 border-t border-slate-800/80 text-[10px] text-slate-400">
              <span className="font-semibold text-slate-300">EXECUTION PIPELINE TRACE:</span>
              {autoResponse.execution_trace?.map((step: any) => (
                <div key={step.step} className="flex items-center space-x-2">
                  <span className="w-4 h-4 rounded-full bg-emerald-500/20 text-emerald-300 flex items-center justify-center text-[9px]">
                    {step.step}
                  </span>
                  <span className="text-slate-300">{step.task}:</span>
                  <span className="text-slate-500 truncate">{step.output_summary}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 2. Manual Search Parameters Form */}
      <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
          <div className="flex items-center space-x-2">
            <Satellite className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider font-mono">
              Copernicus CDSE Satellite Catalog Search
            </h3>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            STAC Spatial Scene Discovery
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
          {/* Location Input */}
          <div className="space-y-1.5">
            <label className="text-slate-400 font-medium font-mono text-[11px] flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5 text-cyan-400" /> LOCATION / TARGET AOI
            </label>
            <input
              type="text"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              placeholder="e.g. Chennai, Tamil Nadu"
              className="w-full bg-slate-950 border border-slate-700/80 rounded-md px-3 py-2 text-slate-200 text-xs focus:outline-none focus:border-emerald-500 transition-colors"
            />
          </div>

          {/* Date Range */}
          <div className="space-y-1.5">
            <label className="text-slate-400 font-medium font-mono text-[11px] flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-cyan-400" /> TEMPORAL WINDOW
            </label>
            <div className="flex items-center space-x-2">
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-1/2 bg-slate-950 border border-slate-700/80 rounded-md px-2.5 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-emerald-500"
              />
              <span className="text-slate-500 text-xs">→</span>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-1/2 bg-slate-950 border border-slate-700/80 rounded-md px-2.5 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-emerald-500"
              />
            </div>
          </div>

          {/* Cloud Cover Slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-[11px] font-mono">
              <span className="text-slate-400 flex items-center gap-1.5">
                <CloudSun className="w-3.5 h-3.5 text-cyan-400" /> MAX CLOUD COVER
              </span>
              <span className="text-emerald-400 font-semibold">{cloudCover}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="50"
              value={cloudCover}
              onChange={(e) => setCloudCover(Number(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>0% (Cloud Free)</span>
              <span>50% (Max)</span>
            </div>
          </div>
        </div>

        {/* Sensors & Imagery Type */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-slate-800/60 text-xs">
          {/* Sensors */}
          <div className="space-y-1.5">
            <label className="text-slate-400 font-medium font-mono text-[11px]">
              CONSTELLATIONS & SENSORS
            </label>
            <div className="flex flex-wrap gap-2">
              {["Sentinel-2", "Sentinel-1", "Cartosat-3", "RISAT-1"].map((sensor) => {
                const isChecked = selectedSensors.includes(sensor);
                return (
                  <button
                    key={sensor}
                    onClick={() => toggleSensor(sensor)}
                    className={cn(
                      "px-2.5 py-1 rounded text-xs font-mono transition-all flex items-center space-x-1.5 border",
                      isChecked
                        ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/40"
                        : "bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700"
                    )}
                  >
                    <span className={cn("w-2 h-2 rounded-full", isChecked ? "bg-emerald-400" : "bg-slate-600")} />
                    <span>{sensor}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Imagery Processing Level */}
          <div className="space-y-1.5">
            <label className="text-slate-400 font-medium font-mono text-[11px]">
              PROCESSING LEVEL & PRODUCT TYPE
            </label>
            <select
              value={imageryType}
              onChange={(e) => setImageryType(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700/80 rounded-md px-3 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-emerald-500"
            >
              <option>Level-2A Bottom-of-Atmosphere (BOA) Reflectance</option>
              <option>Level-1C Top-of-Atmosphere (TOA) Orthorectified</option>
              <option>Level-1 GRD SAR Calibrated Gamma0 Backscatter</option>
              <option>Co-Registered Optical + SAR Dual Product</option>
            </select>
          </div>
        </div>

        {/* Find Imagery Button */}
        <div className="pt-2 flex justify-end">
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className="px-5 py-2 rounded-md bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-semibold text-xs flex items-center space-x-2 transition-all shadow-lg shadow-emerald-950/40 disabled:opacity-50"
          >
            {isSearching ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>QUERYING SATELLITE CATALOG...</span>
              </>
            ) : (
              <>
                <Search className="w-3.5 h-3.5" />
                <span>FIND IMAGERY</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* 3. Discovered Candidate Scenes List */}
      {searchResults && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>DISCOVERED CANDIDATE SCENES ({searchResults.length} AVAILABLE TILES)</span>
            <span>RANKED BY MULTI-CRITERIA SCORE</span>
          </div>

          <div className="space-y-3">
            {searchResults.map((scene) => {
              const isPrimary = selectedSceneId === scene.id;
              const isSecondary = selectedSecondarySceneId === scene.id;

              return (
                <div
                  key={scene.id}
                  className={cn(
                    "p-4 rounded-lg border transition-all flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4",
                    isPrimary || isSecondary
                      ? "bg-slate-800/90 border-emerald-500 shadow-md shadow-emerald-950/40"
                      : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
                  )}
                >
                  <div className="flex items-start space-x-3.5">
                    {/* Thumbnail Preview */}
                    {scene.preview_url ? (
                      <img
                        src={scene.preview_url}
                        alt="Thumbnail"
                        className="w-16 h-16 rounded border border-slate-700 object-contain bg-slate-950 shrink-0"
                      />
                    ) : (
                      <div className="w-16 h-16 rounded border border-slate-800 bg-slate-950 flex items-center justify-center shrink-0">
                        <Satellite className="w-6 h-6 text-slate-600" />
                      </div>
                    )}

                    <div className="space-y-1">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-xs font-semibold text-slate-100">{scene.title}</span>
                        {scene.is_selected && (
                          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold">
                            OPTIMAL MATCH
                          </span>
                        )}
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700">
                          {scene.product_type}
                        </span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-amber-400 border border-slate-700">
                          ☁ {scene.cloud_coverage_percentage}% Cloud
                        </span>
                      </div>

                      <div className="text-[11px] text-slate-400 font-mono">
                        ID: {scene.id}
                      </div>

                      <div className="text-[10px] text-slate-500 font-mono">
                        Acquisition: {scene.acquisition_date} • Size: {scene.file_size_formatted} • Overlap: {scene.aoi_overlap_percentage}%
                      </div>

                      {scene.selection_reason && (
                        <div className="text-[10px] text-emerald-400 font-mono pt-0.5">
                          ✓ {scene.selection_reason}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center space-x-2 shrink-0">
                    <button
                      onClick={() => handleApplyScene(scene, false)}
                      className={cn(
                        "px-3 py-1.5 rounded text-xs font-medium transition-all flex items-center space-x-1.5",
                        isPrimary
                          ? "bg-emerald-500 text-slate-950 font-semibold"
                          : "bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                      )}
                    >
                      {isPrimary ? (
                        <>
                          <Check className="w-3.5 h-3.5" />
                          <span>Primary (T1)</span>
                        </>
                      ) : (
                        <span>Set as Primary</span>
                      )}
                    </button>

                    <button
                      onClick={() => handleApplyScene(scene, true)}
                      className={cn(
                        "px-3 py-1.5 rounded text-xs font-medium transition-all flex items-center space-x-1.5",
                        isSecondary
                          ? "bg-cyan-500 text-slate-950 font-semibold"
                          : "bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                      )}
                    >
                      {isSecondary ? (
                        <>
                          <Check className="w-3.5 h-3.5" />
                          <span>Secondary (T2)</span>
                        </>
                      ) : (
                        <span>Set as Secondary</span>
                      )}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
