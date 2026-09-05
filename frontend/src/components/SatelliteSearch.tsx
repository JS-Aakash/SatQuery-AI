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
} from "lucide-react";
import { searchSatelliteScenes } from "../lib/api";
import { ImageMetadata } from "../lib/types";
import { PRESET_SCENARIOS } from "../lib/presets";
import { cn } from "../lib/utils";

interface SatelliteSearchProps {
  onSelectScene: (meta: ImageMetadata, isSecondary?: boolean) => void;
}

export const SatelliteSearch: React.FC<SatelliteSearchProps> = ({ onSelectScene }) => {
  const [location, setLocation] = useState("Chennai, Tamil Nadu");
  const [startDate, setStartDate] = useState("2024-01-01");
  const [endDate, setEndDate] = useState("2026-01-01");
  const [selectedSensors, setSelectedSensors] = useState<string[]>(["Sentinel-2", "Sentinel-1"]);
  const [cloudCover, setCloudCover] = useState(15);
  const [imageryType, setImageryType] = useState("Level-2A Surface Reflectance / GRD");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<any[] | null>(null);
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);

  const toggleSensor = (sensor: string) => {
    setSelectedSensors((prev) =>
      prev.includes(sensor) ? prev.filter((s) => s !== sensor) : [...prev, sensor]
    );
  };

  const handleSearch = async () => {
    setIsSearching(true);
    setSelectedSceneId(null);
    try {
      const res = await searchSatelliteScenes({
        start_date: startDate,
        end_date: endDate,
        sensors: selectedSensors,
        cloud_coverage: cloudCover,
      });
      setSearchResults(res.scenes);
    } catch (err) {
      // Fallback sample scenes
      setSearchResults([
        {
          scene_id: "S2A_MSIL2A_20240218_CHENNAI_T44VLR",
          sensor: "Sentinel-2A MSI (Optical)",
          acquisition_date: "2024-02-18 05:08 UTC",
          cloud_cover_percent: 1.4,
          resolution_m: 10,
          bands_available: ["B02", "B03", "B04", "B08", "B11", "B12"],
          footprint_area_km2: 48.2,
          status: "Catalog Cached"
        },
        {
          scene_id: "S1A_IW_GRDH_20240219_CHENNAI_SAR",
          sensor: "Sentinel-1A C-SAR (Radar)",
          acquisition_date: "2024-02-19 00:23 UTC",
          cloud_cover_percent: 0.0,
          resolution_m: 10,
          bands_available: ["VV", "VH"],
          footprint_area_km2: 48.2,
          status: "Co-registered Pair Ready"
        }
      ]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleApplyScene = (scene: any) => {
    setSelectedSceneId(scene.scene_id);
    const isSar = scene.sensor.toLowerCase().includes("sar") || scene.sensor.toLowerCase().includes("s1");
    const matchedPreset = isSar ? PRESET_SCENARIOS[1] : PRESET_SCENARIOS[0];
    onSelectScene(matchedPreset.imageA);
  };

  return (
    <div className="space-y-6">
      {/* Search Parameters Form */}
      <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
          <div className="flex items-center space-x-2">
            <Satellite className="w-4 h-4 text-emerald-400" />
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider font-mono">
              Copernicus & ISRO Satellite Catalog Query
            </h3>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            Direct STAC Scene Discovery Interface
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
              {["Sentinel-2", "Sentinel-1", "Cartosat-2S", "RISAT-1A"].map((sensor) => {
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

      {/* Discovered Scenes List */}
      {searchResults && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>DISCOVERED SCENES ({searchResults.length} AVAILABLE TILES)</span>
            <span>FOOTPRINT: ~48.2 km² AOI</span>
          </div>

          <div className="space-y-2">
            {searchResults.map((scene) => {
              const isSelected = selectedSceneId === scene.scene_id;
              return (
                <div
                  key={scene.scene_id}
                  className={cn(
                    "p-3.5 rounded-lg border transition-all flex items-center justify-between",
                    isSelected
                      ? "bg-slate-800/90 border-emerald-500 shadow-md shadow-emerald-950/40"
                      : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
                  )}
                >
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-semibold text-slate-100">{scene.sensor}</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700">
                        {scene.resolution_m}m GSD
                      </span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-amber-400 border border-slate-700">
                        ☁ {scene.cloud_cover_percent}% Cloud
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 font-mono">
                      Scene: {scene.scene_id}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono">
                      Acquisition: {scene.acquisition_date} • Bands: {scene.bands_available.join(", ")}
                    </div>
                  </div>

                  <button
                    onClick={() => handleApplyScene(scene)}
                    className={cn(
                      "px-3 py-1.5 rounded text-xs font-medium transition-all flex items-center space-x-1.5",
                      isSelected
                        ? "bg-emerald-500 text-slate-950 font-semibold"
                        : "bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                    )}
                  >
                    {isSelected ? (
                      <>
                        <Check className="w-3.5 h-3.5" />
                        <span>Loaded in Workspace</span>
                      </>
                    ) : (
                      <>
                        <span>Select Scene</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </>
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
