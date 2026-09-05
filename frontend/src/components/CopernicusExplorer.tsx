"use client";

import React, { useState, useEffect } from "react";
import {
  Search,
  MapPin,
  Calendar,
  Cloud,
  Sliders,
  Sparkles,
  Activity,
  CheckCircle2,
  AlertCircle,
  Clock,
  Layers,
  TrendingUp,
  Brain,
  Download,
  Eye,
  RefreshCw,
  Satellite,
  Compass,
  Zap,
  ArrowRight,
  ShieldCheck,
  Split,
  ChevronRight,
  Maximize2
} from "lucide-react";
import SatelliteMap from "./SatelliteMap";
import {
  geocodeCopernicusLocation,
  searchCopernicusCatalog,
  processCopernicusAnalysis,
  processCopernicusChangeDetection,
  analyzeCopernicusWithAI
} from "../lib/api";
import {
  CopernicusAOI,
  CopernicusObservationItem,
  ProcessAnalysisResponseData,
  TemporalChangeResponseData,
  SpectralStatisticsData,
  GeocodeLocationResult,
  GroundingBox
} from "../lib/types";

const POPULAR_LOCATIONS = [
  { name: "Perundurai, TN", query: "Perundurai" },
  { name: "Erode, TN", query: "Erode" },
  { name: "Coimbatore, TN", query: "Coimbatore" },
  { name: "Chennai Port, TN", query: "Chennai" },
  { name: "Bengaluru, KA", query: "Bengaluru" },
];

export default function CopernicusExplorer() {
  // 1. Location & AOI State
  const [searchQuery, setSearchQuery] = useState<string>("Perundurai");
  const [locationName, setLocationName] = useState<string>("Perundurai, Tamil Nadu, India");
  const [mapCenter, setMapCenter] = useState<[number, number]>([11.2750, 77.5833]);
  const [aoiAreaKm2, setAoiAreaKm2] = useState<number>(5.42);
  const [aoi, setAoi] = useState<CopernicusAOI>({
    type: "Polygon",
    coordinates: [
      [
        [77.5300, 11.2300],
        [77.6350, 11.2300],
        [77.6350, 11.3200],
        [77.5300, 11.3200],
        [77.5300, 11.2300],
      ],
    ],
    bbox: [77.5300, 11.2300, 77.6350, 11.3200],
  });

  // 2. Catalog Search State
  const [startDate, setStartDate] = useState<string>("2024-01-01");
  const [endDate, setEndDate] = useState<string>("2026-08-31");
  const [maxCloud, setMaxCloud] = useState<number>(20);
  const [collection, setCollection] = useState<string>("sentinel-2-l2a");
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [observations, setObservations] = useState<CopernicusObservationItem[]>([]);
  const [selectedObsId, setSelectedObsId] = useState<string>("");
  const [searchStatusMsg, setSearchStatusMsg] = useState<string>("");

  // 3. Spectral Processing State
  const [analysisType, setAnalysisType] = useState<string>("NDBI");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processResult, setProcessResult] = useState<ProcessAnalysisResponseData | null>(null);
  const [processError, setProcessError] = useState<string | null>(null);

  // 4. Temporal Change Detection State
  const [isChangeMode, setIsChangeMode] = useState<boolean>(true);
  const [beforeObsId, setBeforeObsId] = useState<string>("");
  const [afterObsId, setAfterObsId] = useState<string>("");
  const [isComputingChange, setIsComputingChange] = useState<boolean>(false);
  const [changeResult, setChangeResult] = useState<TemporalChangeResponseData | null>(null);
  const [activeLayerMode, setActiveLayerMode] = useState<"auto" | "before" | "after" | "diff">("diff");

  // 5. AI Reasoning Layer State
  const [aiQuestion, setAiQuestion] = useState<string>("How much has building and industrial infrastructure increased?");
  const [isAiAnalyzing, setIsAiAnalyzing] = useState<boolean>(false);
  const [aiAnalysisResult, setAiAnalysisResult] = useState<any>(null);
  const [groundingBoxes, setGroundingBoxes] = useState<GroundingBox[]>([]);

  // Initial geocoding & catalog search on mount
  useEffect(() => {
    handleGeocodeAndSearch("Perundurai");
  }, []);

  const handleGeocodeAndSearch = async (locQuery: string) => {
    try {
      const geo: GeocodeLocationResult = await geocodeCopernicusLocation(locQuery);
      setMapCenter([geo.lat, geo.lon]);
      setLocationName(geo.display_name || geo.name);
      const dLat = 0.04;
      const dLon = 0.05;
      const newCoords = [
        [geo.lon - dLon, geo.lat - dLat],
        [geo.lon + dLon, geo.lat - dLat],
        [geo.lon + dLon, geo.lat + dLat],
        [geo.lon - dLon, geo.lat + dLat],
        [geo.lon - dLon, geo.lat - dLat],
      ];
      const newAoi: CopernicusAOI = {
        type: "Polygon",
        coordinates: [newCoords],
        bbox: [geo.lon - dLon, geo.lat - dLat, geo.lon + dLon, geo.lat + dLat],
      };
      setAoi(newAoi);
      setAoiAreaKm2(5.42);

      // Perform Catalog Search for this AOI
      await performCatalogSearch(newAoi);
    } catch (err: any) {
      console.warn("Geocoding failed, using local presets:", err);
    }
  };

  const performCatalogSearch = async (targetAoi: CopernicusAOI) => {
    setIsSearching(true);
    setSearchStatusMsg("Querying Copernicus STAC Catalog...");
    try {
      const res = await searchCopernicusCatalog({
        aoi: targetAoi,
        start_date: startDate,
        end_date: endDate,
        max_cloud_coverage: maxCloud,
        collection: collection,
        limit: 10,
      });

      setObservations(res.observations || []);
      setAoiAreaKm2(res.aoi_area_sq_km || 5.42);
      setSearchStatusMsg(res.status_message || `Found ${res.total_count} observations.`);

      if (res.observations && res.observations.length > 0) {
        setSelectedObsId(res.observations[0].id);
        if (res.observations.length >= 2) {
          setBeforeObsId(res.observations[res.observations.length - 1].id);
          setAfterObsId(res.observations[0].id);
        }
      }
    } catch (err: any) {
      setSearchStatusMsg(`Search failed: ${err.message}`);
    } finally {
      setIsSearching(false);
    }
  };

  const handleAOIChange = (newAoi: CopernicusAOI, areaKm2: number) => {
    setAoi(newAoi);
    setAoiAreaKm2(areaKm2);
  };

  const handleExecuteProcessing = async (typeOverride?: string) => {
    const selectedType = typeOverride || analysisType;
    if (!selectedObsId) {
      setProcessError("Please select a satellite observation candidate first.");
      return;
    }

    setIsProcessing(true);
    setProcessError(null);
    try {
      const res: ProcessAnalysisResponseData = await processCopernicusAnalysis({
        aoi: aoi,
        observation_id: selectedObsId,
        analysis_type: selectedType,
        collection: collection,
        resolution_m: 10.0,
      });

      setProcessResult(res);
      setAoiAreaKm2(res.aoi_area_sq_km);
      setActiveLayerMode("auto");
      setGroundingBoxes([]);
    } catch (err: any) {
      setProcessError(err.message || "Failed to process imagery via Sentinel Hub Processing API.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExecuteChangeDetection = async () => {
    if (!beforeObsId || !afterObsId) {
      setProcessError("Please select both Before and After observations.");
      return;
    }

    setIsComputingChange(true);
    setProcessError(null);
    try {
      const res: TemporalChangeResponseData = await processCopernicusChangeDetection({
        aoi: aoi,
        before_observation_id: beforeObsId,
        after_observation_id: afterObsId,
        analysis_type: analysisType,
      });

      setChangeResult(res);
      setActiveLayerMode("diff");
      if (res.bounding_boxes && res.bounding_boxes.length > 0) {
        setGroundingBoxes(res.bounding_boxes as any);
      }
    } catch (err: any) {
      setProcessError(err.message || "Bi-temporal change detection failed.");
    } finally {
      setIsComputingChange(false);
    }
  };

  const handleTriggerAIEvaluation = async () => {
    if (!processResult && !changeResult) {
      setProcessError("Please process satellite imagery before requesting AI agent analysis.");
      return;
    }

    setIsAiAnalyzing(true);
    try {
      const imgUrl = processResult?.image_url || changeResult?.diff_image_url || "";
      const stats = processResult?.statistics || {};
      const changeStats = changeResult?.change_statistics || {};

      const res = await analyzeCopernicusWithAI({
        query: aiQuestion,
        image_url: imgUrl,
        analysis_type: analysisType,
        statistics: stats,
        change_statistics: changeStats,
        location_name: locationName,
        aoi_area_sq_km: aoiAreaKm2,
        temporal_dates: changeResult ? [changeResult.before_date, changeResult.after_date] : undefined,
      });

      setAiAnalysisResult(res);
      if (res.bounding_boxes && res.bounding_boxes.length > 0) {
        setGroundingBoxes(res.bounding_boxes);
      }
    } catch (err: any) {
      setProcessError(err.message || "AI Analysis failed.");
    } finally {
      setIsAiAnalyzing(false);
    }
  };

  // Determine active overlay image and legend for SatelliteMap
  let activeOverlayUrl: string | undefined = undefined;
  let activeLegend: Record<string, string> | undefined = undefined;
  let activeTitle: string | undefined = undefined;

  if (isChangeMode && changeResult) {
    if (activeLayerMode === "before") {
      activeOverlayUrl = changeResult.before_image_url;
      activeTitle = `Date 1 (${changeResult.before_date.split("T")[0]})`;
    } else if (activeLayerMode === "after") {
      activeOverlayUrl = changeResult.after_image_url;
      activeTitle = `Date 2 (${changeResult.after_date.split("T")[0]})`;
    } else {
      activeOverlayUrl = changeResult.diff_image_url;
      activeLegend = changeResult.legend;
      activeTitle = `Change Map (Δ ${changeResult.analysis_type})`;
    }
  } else if (processResult) {
    activeOverlayUrl = processResult.image_url;
    activeLegend = processResult.legend;
    activeTitle = processResult.analysis_type;
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* 1. Top Header & Authentication Status Banner */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-emerald-950/40 to-slate-900 border border-emerald-500/30 shadow-2xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2.5">
            <Satellite className="w-5 h-5 text-emerald-400 animate-pulse" />
            <h2 className="text-lg font-bold text-slate-100 tracking-tight">
              Copernicus Data Space & Sentinel Hub Analysis Module
            </h2>
            <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-[10px] font-mono">
              CDSE OAUTH2 ACTIVE
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real Sentinel-2 L2A bottom-of-atmosphere observation discovery, multi-date visual comparison, authentic pixel statistics, and AI vision reasoning.
          </p>
        </div>

        <div className="flex items-center space-x-2 text-xs font-mono text-slate-300 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Native 10m/px Ground Resolution</span>
        </div>
      </div>

      {/* 2. Location Search & AOI Setup Card */}
      <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
            <MapPin className="w-4 h-4 text-cyan-400" />
            <span>1. Location Geocoding & AOI Definition:</span>
            <span className="text-cyan-300 font-mono text-xs">{locationName}</span>
          </div>

          {/* Quick Predefined Location Chips */}
          <div className="flex items-center space-x-1.5 overflow-x-auto">
            <span className="text-[11px] font-mono text-slate-500 uppercase mr-1">PRESETS:</span>
            {POPULAR_LOCATIONS.map((loc) => (
              <button
                key={loc.query}
                onClick={() => {
                  setSearchQuery(loc.query);
                  handleGeocodeAndSearch(loc.query);
                }}
                className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                  searchQuery.toLowerCase() === loc.query.toLowerCase()
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold"
                    : "bg-slate-800/80 text-slate-400 hover:text-slate-200 border border-slate-700/60"
                }`}
              >
                {loc.name}
              </button>
            ))}
          </div>
        </div>

        {/* Search input with Nominatim query trigger */}
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleGeocodeAndSearch(searchQuery);
              }}
              placeholder="Search place name (e.g. Perundurai, Erode, Coimbatore, Chennai) or enter lat, lon..."
              className="w-full pl-10 pr-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors font-mono"
            />
          </div>
          <button
            onClick={() => handleGeocodeAndSearch(searchQuery)}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-900 font-bold text-xs rounded-lg transition-colors flex items-center space-x-1.5 shadow-lg shadow-cyan-900/20"
          >
            <Compass className="w-3.5 h-3.5" />
            <span>Search Map</span>
          </button>
        </div>

        {/* Multi-Date Layer Switcher on Top of Map if Change Result Available */}
        {isChangeMode && changeResult && (
          <div className="flex items-center justify-between p-2.5 bg-slate-950/90 rounded-lg border border-slate-800 text-xs font-mono">
            <div className="flex items-center space-x-2 text-slate-300 font-bold">
              <Eye className="w-4 h-4 text-emerald-400" />
              <span>MAP OVERLAY VIEW:</span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setActiveLayerMode("before")}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  activeLayerMode === "before"
                    ? "bg-blue-600 text-white font-bold"
                    : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
                }`}
              >
                Date 1 (Before: {changeResult.before_date.split("T")[0]})
              </button>
              <button
                onClick={() => setActiveLayerMode("after")}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  activeLayerMode === "after"
                    ? "bg-blue-600 text-white font-bold"
                    : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
                }`}
              >
                Date 2 (After: {changeResult.after_date.split("T")[0]})
              </button>
              <button
                onClick={() => setActiveLayerMode("diff")}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  activeLayerMode === "diff"
                    ? "bg-rose-600 text-white font-bold"
                    : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
                }`}
              >
                Delta Change Map (Difference)
              </button>
            </div>
          </div>
        )}

        {/* Interactive Satellite Map Component with Grounding Boxes */}
        <SatelliteMap
          center={mapCenter}
          aoi={aoi}
          onAOIChange={handleAOIChange}
          areaKm2={aoiAreaKm2}
          overlayImageUrl={activeOverlayUrl}
          overlayLegend={activeLegend}
          overlayTitle={activeTitle}
          groundingBoxes={groundingBoxes}
        />
      </div>

      {/* 3. Real Copernicus Catalog Search & Candidates Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Search Filter Controls */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4 lg:col-span-1">
          <div className="flex items-center space-x-2 text-sm font-bold text-slate-200 pb-2 border-b border-slate-800">
            <Satellite className="w-4 h-4 text-emerald-400" />
            <span>2. Catalog Search Criteria</span>
          </div>

          <div className="space-y-3.5 text-xs font-mono">
            {/* Satellite Collection */}
            <div>
              <label className="text-slate-400 block mb-1">SATELLITE / COLLECTION</label>
              <select
                value={collection}
                onChange={(e) => setCollection(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-emerald-500"
              >
                <option value="sentinel-2-l2a">Sentinel-2 L2A (10m Optical / Multispectral)</option>
                <option value="sentinel-1-grd">Sentinel-1 GRD (SAR Interferometric)</option>
              </select>
            </div>

            {/* Date Range: Supports Multi-Year Comparison */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-slate-400 block mb-1">START DATE (T1)</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="text-slate-400 block mb-1">END DATE (T2)</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>

            {/* Cloud Coverage Slider */}
            <div>
              <div className="flex justify-between text-slate-400 mb-1">
                <span>MAX CLOUD COVER</span>
                <span className="text-emerald-400 font-bold">{maxCloud}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={maxCloud}
                onChange={(e) => setMaxCloud(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
              />
            </div>

            {/* Execute Search Button */}
            <button
              onClick={() => performCatalogSearch(aoi)}
              disabled={isSearching}
              className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-slate-950 font-bold rounded-lg transition-colors flex items-center justify-center space-x-2 shadow-lg shadow-emerald-900/20"
            >
              {isSearching ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              <span>{isSearching ? "Searching CDSE STAC..." : "Discover Satellite Scenes"}</span>
            </button>

            {searchStatusMsg && (
              <p className="text-[11px] text-slate-400 italic text-center">{searchStatusMsg}</p>
            )}
          </div>
        </div>

        {/* Right Column: Candidate Observations List */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-3 lg:col-span-2 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span>Candidate Satellite Observations ({observations.length})</span>
              </div>
              <span className="text-xs font-mono text-slate-400">Resolution: 10m L2A</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 mt-3 max-h-[220px] overflow-y-auto pr-1">
              {observations.map((obs) => {
                const isSelected = selectedObsId === obs.id;
                return (
                  <div
                    key={obs.id}
                    onClick={() => setSelectedObsId(obs.id)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all flex flex-col justify-between ${
                      isSelected
                        ? "bg-emerald-500/15 border-emerald-500/50 shadow-md"
                        : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="font-bold text-slate-200">{obs.acquisition_date.split("T")[0]}</span>
                        <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                          {obs.platform}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mt-1.5">
                        <span>Cloud: {obs.cloud_coverage_percent}%</span>
                        <span>10m BOA L2A</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-2 mt-2 border-t border-slate-800/80 text-[10px] font-mono">
                      <span className="text-slate-500 truncate max-w-[120px]">{obs.id.split("_")[0]}</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedObsId(obs.id);
                          handleExecuteProcessing("TRUE_COLOR");
                        }}
                        className="text-cyan-400 hover:text-cyan-300 flex items-center space-x-1"
                        title="Preview true-color raster on map"
                      >
                        <Eye className="w-3 h-3" />
                        <span>Quick Preview</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

          </div>
        </div>
      </div>

      {/* 4. Analysis Execution Modes (Single Date vs Bi-Temporal Change) */}
      <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
            <Sliders className="w-4 h-4 text-amber-400" />
            <span>3. Satellite Processing & Change Detection Engine</span>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsChangeMode(false)}
              className={`px-3 py-1 rounded-lg text-xs font-mono transition-colors ${
                !isChangeMode
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold"
                  : "bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800"
              }`}
            >
              Single Date Analysis
            </button>
            <button
              onClick={() => setIsChangeMode(true)}
              className={`px-3 py-1 rounded-lg text-xs font-mono transition-colors ${
                isChangeMode
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold"
                  : "bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800"
              }`}
            >
              Bi-Temporal Change Mode
            </button>
          </div>
        </div>

        {/* Analysis Type Selector */}
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
            {[
              { id: "NDBI", name: "NDBI (Built-up / Urban)", desc: "(SWIR - NIR) / (SWIR + NIR)" },
              { id: "NDVI", name: "NDVI (Vegetation)", desc: "(NIR - Red) / (NIR + Red)" },
              { id: "NDWI", name: "NDWI (Water)", desc: "(Green - NIR) / (Green + NIR)" },
              { id: "TRUE_COLOR", name: "True Color RGB", desc: "Natural Human Vision (B04, B03, B02)" },
              { id: "FALSE_COLOR_NIR", name: "False Color NIR", desc: "Chlorophyll Contrast (B08, B04, B03)" },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setAnalysisType(item.id);
                  if (!isChangeMode) handleExecuteProcessing(item.id);
                }}
                className={`p-3 rounded-xl border text-left transition-all ${
                  analysisType === item.id
                    ? "bg-amber-500/15 border-amber-500/50 shadow-lg shadow-amber-950/20"
                    : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
                }`}
              >
                <div className="font-bold text-xs text-slate-200">{item.name}</div>
                <div className="text-[10px] font-mono text-slate-500 mt-1">{item.desc}</div>
              </button>
            ))}
          </div>

          {!isChangeMode ? (
            /* Single Date Controls */
            <div className="flex items-center justify-between pt-2">
              <div className="text-xs font-mono text-slate-400">
                Selected: <span className="text-amber-400 font-bold">{analysisType}</span> on observation{" "}
                <span className="text-cyan-400">{selectedObsId ? selectedObsId.split("_")[2] || selectedObsId : "None"}</span>
              </div>

              <button
                onClick={() => handleExecuteProcessing()}
                disabled={isProcessing || !selectedObsId}
                className="px-5 py-2.5 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 font-bold text-xs rounded-lg transition-colors flex items-center space-x-2 shadow-lg shadow-amber-900/20"
              >
                {isProcessing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                <span>{isProcessing ? "Processing via Sentinel Hub API..." : "Process Selected AOI"}</span>
              </button>
            </div>
          ) : (
            /* Bi-Temporal Change Controls */
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                  <label className="text-xs font-mono text-slate-400 block font-bold">BEFORE OBSERVATION (T1 - BASELINE)</label>
                  <select
                    value={beforeObsId}
                    onChange={(e) => setBeforeObsId(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs font-mono text-slate-200"
                  >
                    {observations.map((o) => (
                      <option key={`before-${o.id}`} value={o.id}>
                        {o.acquisition_date.split("T")[0]} ({o.cloud_coverage_percent}% Cloud) - {o.platform}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                  <label className="text-xs font-mono text-slate-400 block font-bold">AFTER OBSERVATION (T2 - MONITORING)</label>
                  <select
                    value={afterObsId}
                    onChange={(e) => setAfterObsId(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs font-mono text-slate-200"
                  >
                    {observations.map((o) => (
                      <option key={`after-${o.id}`} value={o.id}>
                        {o.acquisition_date.split("T")[0]} ({o.cloud_coverage_percent}% Cloud) - {o.platform}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                onClick={handleExecuteChangeDetection}
                disabled={isComputingChange || !beforeObsId || !afterObsId}
                className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 text-slate-950 font-bold text-xs rounded-lg transition-colors flex items-center justify-center space-x-2 shadow-lg shadow-rose-900/20"
              >
                {isComputingChange ? <RefreshCw className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
                <span>{isComputingChange ? "Calculating Temporal Pixel Delta..." : "Compute Bi-Temporal Change Map"}</span>
              </button>
            </div>
          )}

          {processError && (
            <div className="p-3 bg-rose-950/60 border border-rose-500/40 text-rose-300 text-xs font-mono rounded-lg flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
              <span>{processError}</span>
            </div>
          )}
        </div>
      </div>

      {/* 5. Bi-Temporal Visual Comparison Gallery (Side-by-Side 3-Panel Inspection) */}
      {isChangeMode && changeResult && (
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
              <Split className="w-4 h-4 text-cyan-400" />
              <span>Multi-Date Visual Comparison ({changeResult.before_date.split("T")[0]} vs {changeResult.after_date.split("T")[0]})</span>
            </div>
            <span className="text-xs font-mono text-cyan-400">Click any image to view on main map</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Panel 1: Date 1 (Before) */}
            <div
              onClick={() => setActiveLayerMode("before")}
              className={`p-3.5 rounded-xl border cursor-pointer transition-all space-y-2 ${
                activeLayerMode === "before"
                  ? "bg-blue-950/40 border-blue-500/60 shadow-lg shadow-blue-950/20"
                  : "bg-slate-950 border-slate-800 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-blue-400">DATE 1 (BEFORE)</span>
                <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded">
                  {changeResult.before_date.split("T")[0]}
                </span>
              </div>
              <div className="aspect-square rounded-lg overflow-hidden bg-slate-900 border border-slate-800 relative flex items-center justify-center">
                {changeResult.before_image_url ? (
                  <img
                    src={changeResult.before_image_url}
                    alt="Before Observation"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-xs font-mono text-slate-500">Loading Preview...</span>
                )}
              </div>
              <p className="text-[11px] font-mono text-slate-400 text-center">Baseline Surface Reflectance</p>
            </div>

            {/* Panel 2: Date 2 (After) */}
            <div
              onClick={() => setActiveLayerMode("after")}
              className={`p-3.5 rounded-xl border cursor-pointer transition-all space-y-2 ${
                activeLayerMode === "after"
                  ? "bg-blue-950/40 border-blue-500/60 shadow-lg shadow-blue-950/20"
                  : "bg-slate-950 border-slate-800 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-blue-400">DATE 2 (AFTER)</span>
                <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded">
                  {changeResult.after_date.split("T")[0]}
                </span>
              </div>
              <div className="aspect-square rounded-lg overflow-hidden bg-slate-900 border border-slate-800 relative flex items-center justify-center">
                {changeResult.after_image_url ? (
                  <img
                    src={changeResult.after_image_url}
                    alt="After Observation"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-xs font-mono text-slate-500">Loading Preview...</span>
                )}
              </div>
              <p className="text-[11px] font-mono text-slate-400 text-center">Monitoring Surface Reflectance</p>
            </div>

            {/* Panel 3: Delta Difference Map */}
            <div
              onClick={() => setActiveLayerMode("diff")}
              className={`p-3.5 rounded-xl border cursor-pointer transition-all space-y-2 ${
                activeLayerMode === "diff"
                  ? "bg-rose-950/40 border-rose-500/60 shadow-lg shadow-rose-950/20"
                  : "bg-slate-950 border-slate-800 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-rose-400">DELTA CHANGE MAP</span>
                <span className="text-[10px] font-mono text-emerald-400 font-bold">
                  Mean Δ: {changeResult.mean_change}
                </span>
              </div>
              <div className="aspect-square rounded-lg overflow-hidden bg-slate-900 border border-slate-800 relative flex items-center justify-center">
                {changeResult.diff_image_url ? (
                  <img
                    src={changeResult.diff_image_url}
                    alt="Delta Change Map"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-xs font-mono text-slate-500">Generating Delta Map...</span>
                )}
              </div>
              <p className="text-[11px] font-mono text-slate-400 text-center">Colorized Pixel Shift (Green=Growth, Red=Loss)</p>
            </div>
          </div>

          {/* Contextual Human-Understandable Metric Tiles */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono pt-2">
            <div className="p-3 bg-slate-950 rounded-lg border border-emerald-500/30 text-center">
              <span className="text-[11px] text-emerald-400 block font-bold uppercase truncate">
                {changeResult.improved_label || "INCREASE / EXPANSION"}
              </span>
              <span className="text-xl font-bold text-emerald-300">
                {changeResult.improved_percent}%
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                +{changeResult.improved_area_km2 || ((changeResult.improved_percent / 100) * changeResult.aoi_area_sq_km).toFixed(2)} km²
              </span>
            </div>

            <div className="p-3 bg-slate-950 rounded-lg border border-slate-700 text-center">
              <span className="text-[11px] text-slate-400 block font-bold uppercase truncate">
                {changeResult.stable_label || "STABLE / UNCHANGED"}
              </span>
              <span className="text-xl font-bold text-slate-200">
                {changeResult.stable_percent}%
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                {changeResult.stable_area_km2 || ((changeResult.stable_percent / 100) * changeResult.aoi_area_sq_km).toFixed(2)} km²
              </span>
            </div>

            <div className="p-3 bg-slate-950 rounded-lg border border-rose-500/30 text-center">
              <span className="text-[11px] text-rose-400 block font-bold uppercase truncate">
                {changeResult.declined_label || "DECREASE / LOSS"}
              </span>
              <span className="text-xl font-bold text-rose-300">
                {changeResult.declined_percent}%
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                {changeResult.declined_area_km2 || ((changeResult.declined_percent / 100) * changeResult.aoi_area_sq_km).toFixed(2)} km²
              </span>
            </div>
          </div>

          {/* Natural Language Summary Card */}
          {changeResult.human_summary && (
            <div className="p-3.5 bg-slate-950 rounded-lg border border-slate-800 text-xs text-slate-300 font-sans leading-relaxed">
              <span className="font-mono text-emerald-400 font-bold mr-1">Summary:</span>
              {changeResult.human_summary}
            </div>
          )}
        </div>
      )}

      {/* 6. Single-Date Statistical Breakdown Card */}
      {!isChangeMode && processResult?.statistics && (
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span>Calculated Spatial Statistics ({processResult.analysis_type})</span>
            </div>
            <div className="text-xs font-mono text-cyan-400">
              AOI: {processResult.aoi_area_sq_km} km² • Res: 10m/px • Pixels: {processResult.statistics.valid_pixel_count.toLocaleString()}
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block">MEAN {processResult.analysis_type}</span>
              <span className="text-lg font-bold text-emerald-400">{processResult.statistics.mean}</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block">MINIMUM</span>
              <span className="text-lg font-bold text-amber-400">{processResult.statistics.min}</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block">MAXIMUM</span>
              <span className="text-lg font-bold text-cyan-400">{processResult.statistics.max}</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block">STD DEVIATION</span>
              <span className="text-lg font-bold text-slate-200">{processResult.statistics.std_dev}</span>
            </div>
          </div>

          <div className="space-y-2 pt-2">
            <div className="text-xs font-mono text-slate-400 uppercase">SURFACE CLASSIFICATION DISTRIBUTION</div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs font-mono">
              {Object.entries(processResult.statistics.class_breakdown).map(([label, pct]) => (
                <div key={label} className="p-2.5 bg-slate-950 rounded-lg border border-slate-800 flex items-center justify-between">
                  <span className="text-slate-300 truncate max-w-[200px]">{label}</span>
                  <span className="text-emerald-400 font-bold ml-2">{pct}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 7. AI Vision-Language Agent Integration Card with Natural Language Querying */}
      <div className="p-5 rounded-xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/30 shadow-2xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
            <Brain className="w-4 h-4 text-indigo-400" />
            <span>4. Natural Language AI Remote Sensing Intelligence</span>
          </div>
          <span className="text-xs font-mono text-indigo-400 bg-indigo-500/20 px-2 py-0.5 rounded border border-indigo-500/30">
            GeoChat-7B RS-VLM & NLP Engine
          </span>
        </div>

        {/* Quick query suggestion chips */}
        <div className="flex items-center space-x-1.5 overflow-x-auto text-xs font-mono">
          <span className="text-slate-500 text-[11px] uppercase mr-1">EXAMPLES:</span>
          {[
            "How much has building increased?",
            "How much has vegetation changed?",
            "What is the surface water dynamics?",
            "Provide full land-cover summary",
          ].map((prompt) => (
            <button
              key={prompt}
              onClick={() => {
                setAiQuestion(prompt);
              }}
              className="px-2.5 py-1 bg-slate-950 hover:bg-indigo-950 text-slate-300 hover:text-indigo-200 rounded border border-slate-800 text-[11px] transition-colors whitespace-nowrap"
            >
              {prompt}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <input
            type="text"
            value={aiQuestion}
            onChange={(e) => setAiQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleTriggerAIEvaluation();
            }}
            placeholder="Ask natural language question (e.g. 'How much has building increased?')..."
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={handleTriggerAIEvaluation}
            disabled={isAiAnalyzing || (!processResult && !changeResult)}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-slate-950 font-bold text-xs rounded-lg transition-colors flex items-center space-x-1.5 shadow-lg shadow-indigo-900/20"
          >
            {isAiAnalyzing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            <span>{isAiAnalyzing ? "Analyzing..." : "Ask AI Agent"}</span>
          </button>
        </div>

        {aiAnalysisResult && (
          <div className="p-4 bg-slate-950/90 rounded-lg border border-indigo-500/40 space-y-3 shadow-xl">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-mono font-bold text-indigo-300 uppercase">
                AI INTELLIGENCE SYNTHESIS ({aiAnalysisResult.model_used})
              </span>
              <span className="text-xs font-mono text-emerald-400">Confidence: {aiAnalysisResult.confidence_formatted}</span>
            </div>

            <p className="text-xs text-slate-200 leading-relaxed font-sans">{aiAnalysisResult.ai_explanation}</p>

            {/* Key Metrics Pills if available */}
            {aiAnalysisResult.key_metrics && Object.keys(aiAnalysisResult.key_metrics).length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-1 font-mono text-xs">
                {Object.entries(aiAnalysisResult.key_metrics).map(([k, v]) => (
                  <div key={k} className="p-2 bg-slate-900 rounded border border-slate-800 flex flex-col">
                    <span className="text-[10px] text-slate-400 uppercase">{k}</span>
                    <span className="text-emerald-400 font-bold mt-0.5">{String(v)}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Grounding Boxes Note */}
            {groundingBoxes.length > 0 && (
              <div className="flex items-center space-x-1.5 text-[11px] font-mono text-amber-300 bg-amber-500/10 px-2.5 py-1.5 rounded border border-amber-500/30">
                <CheckCircle2 className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                <span>
                  {groundingBoxes.length} region(s) delineated and highlighted directly on the interactive map above.
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
