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
  ShieldCheck
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
  GeocodeLocationResult
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
  const [startDate, setStartDate] = useState<string>("2026-08-01");
  const [endDate, setEndDate] = useState<string>("2026-08-31");
  const [maxCloud, setMaxCloud] = useState<number>(20);
  const [collection, setCollection] = useState<string>("sentinel-2-l2a");
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [observations, setObservations] = useState<CopernicusObservationItem[]>([]);
  const [selectedObsId, setSelectedObsId] = useState<string>("");
  const [searchStatusMsg, setSearchStatusMsg] = useState<string>("");

  // 3. Spectral Processing State
  const [analysisType, setAnalysisType] = useState<string>("NDVI");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processResult, setProcessResult] = useState<ProcessAnalysisResponseData | null>(null);
  const [processError, setProcessError] = useState<string | null>(null);

  // 4. Temporal Change Detection State
  const [isChangeMode, setIsChangeMode] = useState<boolean>(false);
  const [beforeObsId, setBeforeObsId] = useState<string>("");
  const [afterObsId, setAfterObsId] = useState<string>("");
  const [isComputingChange, setIsComputingChange] = useState<boolean>(false);
  const [changeResult, setChangeResult] = useState<TemporalChangeResponseData | null>(null);

  // 5. AI Reasoning Layer State
  const [aiQuestion, setAiQuestion] = useState<string>("Analyze vegetation vigor, canopy health, and water absorption dynamics over this AOI.");
  const [isAiAnalyzing, setIsAiAnalyzing] = useState<boolean>(false);
  const [aiAnalysisResult, setAiAnalysisResult] = useState<any>(null);

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
      const stats = processResult?.statistics || changeResult?.change_statistics || {};

      const res = await analyzeCopernicusWithAI({
        query: aiQuestion,
        image_url: imgUrl,
        analysis_type: analysisType,
        statistics: stats,
      });

      setAiAnalysisResult(res);
    } catch (err: any) {
      setProcessError(err.message || "AI Analysis failed.");
    } finally {
      setIsAiAnalyzing(false);
    }
  };

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
            Real Sentinel-2 L2A bottom-of-atmosphere observation discovery, native 10m spectral indices, authentic pixel statistics, and AI vision reasoning.
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
            <span>1. Location Geocoding & AOI Definition</span>
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

        {/* Interactive Satellite Map Component */}
        <SatelliteMap
          center={mapCenter}
          aoi={aoi}
          onAOIChange={handleAOIChange}
          areaKm2={aoiAreaKm2}
          overlayImageUrl={processResult?.image_url || changeResult?.diff_image_url}
          overlayLegend={processResult?.legend || changeResult?.legend}
          overlayTitle={analysisType}
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
                className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200 focus:outline-none focus:border-emerald-500"
              >
                <option value="sentinel-2-l2a">Sentinel-2 L2A (Bottom-of-Atmosphere)</option>
                <option value="sentinel-1-grd">Sentinel-1 SAR C-Band (Ground Range)</option>
              </select>
            </div>

            {/* Date Range */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-slate-400 block mb-1">START DATE</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-slate-400 block mb-1">END DATE</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200 focus:outline-none"
                />
              </div>
            </div>

            {/* Max Cloud Coverage Slider */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-slate-400">MAX CLOUD COVERAGE</label>
                <span className="text-emerald-400 font-bold">{maxCloud}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={maxCloud}
                onChange={(e) => setMaxCloud(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-emerald-500"
              />
            </div>

            <button
              onClick={() => performCatalogSearch(aoi)}
              disabled={isSearching}
              className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-slate-950 font-bold text-xs rounded-lg transition-colors flex items-center justify-center space-x-2 shadow-lg shadow-emerald-900/20 mt-2"
            >
              {isSearching ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              <span>{isSearching ? "Searching Copernicus STAC..." : "Search Satellite Data"}</span>
            </button>
          </div>
        </div>

        {/* Right 2 Columns: Candidate Observations List */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4 lg:col-span-2">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Matching Copernicus Observations ({observations.length})</span>
            </div>
            <span className="text-xs font-mono text-slate-400">{searchStatusMsg}</span>
          </div>

          {observations.length === 0 ? (
            <div className="p-8 text-center border border-dashed border-slate-800 rounded-xl space-y-2">
              <AlertCircle className="w-8 h-8 text-amber-400 mx-auto" />
              <p className="text-xs text-slate-300 font-mono">No matching scenes found for this date & cloud threshold.</p>
              <p className="text-[11px] text-slate-500 font-mono">
                Suggestions: 1) Increase max cloud coverage slider, 2) Expand date range, or 3) Pick another acquisition interval.
              </p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
              {observations.map((obs) => {
                const isSelected = selectedObsId === obs.id;
                return (
                  <div
                    key={obs.id}
                    onClick={() => setSelectedObsId(obs.id)}
                    className={`p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between text-xs font-mono ${
                      isSelected
                        ? "bg-emerald-950/40 border-emerald-500/60 shadow-md shadow-emerald-950/20"
                        : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full border ${isSelected ? "bg-emerald-400 border-emerald-300" : "bg-slate-800 border-slate-700"}`} />
                      <div>
                        <div className="font-bold text-slate-200">
                          {obs.acquisition_date.split("T")[0]} ({obs.platform})
                        </div>
                        <div className="text-[11px] text-slate-500 truncate max-w-[320px]">
                          Tile: {obs.tile_id || "T43Q"} • {obs.id}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-1">
                        <Cloud className={`w-3.5 h-3.5 ${obs.cloud_coverage_percent < 10 ? "text-emerald-400" : "text-amber-400"}`} />
                        <span className="text-slate-300">{obs.cloud_coverage_percent}% Cloud</span>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedObsId(obs.id);
                        }}
                        className={`px-2.5 py-1 rounded text-[11px] font-bold ${
                          isSelected ? "bg-emerald-500 text-slate-950" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}
                      >
                        {isSelected ? "Selected" : "Select"}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* 4. Analysis Selector & Spectral Processing Section */}
      <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
            <Sliders className="w-4 h-4 text-amber-400" />
            <span>3. Spectral Index & Processing Selector</span>
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
              Single-Date Analysis
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

        {!isChangeMode ? (
          /* Single Date Spectral Index Options */
          <div className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
              {[
                { id: "TRUE_COLOR", name: "True Color RGB", desc: "Natural Human Vision (B04, B03, B02)" },
                { id: "FALSE_COLOR_NIR", name: "False Color NIR", desc: "Chlorophyll Contrast (B08, B04, B03)" },
                { id: "NDVI", name: "NDVI (Vegetation)", desc: "(NIR - Red) / (NIR + Red)" },
                { id: "NDWI", name: "NDWI (Water)", desc: "(Green - NIR) / (Green + NIR)" },
                { id: "NDBI", name: "NDBI (Built-up)", desc: "(SWIR - NIR) / (SWIR + NIR)" },
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    setAnalysisType(item.id);
                    handleExecuteProcessing(item.id);
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
          </div>
        ) : (
          /* Bi-Temporal Change Detection Controls */
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                <label className="text-xs font-mono text-slate-400 block font-bold">BEFORE OBSERVATION (T1)</label>
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
                <label className="text-xs font-mono text-slate-400 block font-bold">AFTER OBSERVATION (T2)</label>
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

      {/* 5. Authentic Statistical Results & Spectral Breakdown */}
      {processResult?.statistics && (
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

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-center font-mono">
              <span className="text-[11px] text-slate-400 block">MEAN {processResult.analysis_type}</span>
              <span className="text-lg font-bold text-emerald-400">{processResult.statistics.mean}</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-center font-mono">
              <span className="text-[11px] text-slate-400 block">MINIMUM</span>
              <span className="text-lg font-bold text-amber-400">{processResult.statistics.min}</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-center font-mono">
              <span className="text-[11px] text-slate-400 block">MAXIMUM</span>
              <span className="text-lg font-bold text-cyan-400">{processResult.statistics.max}</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-center font-mono">
              <span className="text-[11px] text-slate-400 block">STD DEVIATION</span>
              <span className="text-lg font-bold text-slate-200">{processResult.statistics.std_dev}</span>
            </div>
          </div>

          {/* Categorical Class Breakdown Distribution */}
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

      {/* 6. Bi-Temporal Change Results Card */}
      {changeResult && (
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
              <TrendingUp className="w-4 h-4 text-rose-400" />
              <span>Bi-Temporal Change Metrics ({changeResult.before_date.split("T")[0]} vs {changeResult.after_date.split("T")[0]})</span>
            </div>
            <span className="text-xs font-mono text-emerald-400 font-bold">Mean Δ: {changeResult.mean_change}</span>
          </div>

          <div className="grid grid-cols-3 gap-3 font-mono">
            <div className="p-3 bg-slate-950 rounded-lg border border-emerald-500/30 text-center">
              <span className="text-[11px] text-emerald-400 block font-bold">IMPROVED / REGROWTH</span>
              <span className="text-xl font-bold text-emerald-300">{changeResult.improved_percent}%</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-700 text-center">
              <span className="text-[11px] text-slate-400 block font-bold">STABLE / UNCHANGED</span>
              <span className="text-xl font-bold text-slate-200">{changeResult.stable_percent}%</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-rose-500/30 text-center">
              <span className="text-[11px] text-rose-400 block font-bold">DECLINED / LOSS</span>
              <span className="text-xl font-bold text-rose-300">{changeResult.declined_percent}%</span>
            </div>
          </div>
        </div>
      )}

      {/* 7. AI Vision-Language Agent Integration Card */}
      <div className="p-5 rounded-xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/30 shadow-2xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
            <Brain className="w-4 h-4 text-indigo-400" />
            <span>4. AI Vision-Language Agentic Synthesis</span>
          </div>
          <span className="text-xs font-mono text-indigo-400 bg-indigo-500/20 px-2 py-0.5 rounded border border-indigo-500/30">
            GeoChat-7B RS-VLM
          </span>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="text"
            value={aiQuestion}
            onChange={(e) => setAiQuestion(e.target.value)}
            placeholder="Ask AI agent to evaluate spectral indices & landscape characteristics..."
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={handleTriggerAIEvaluation}
            disabled={isAiAnalyzing || (!processResult && !changeResult)}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-slate-950 font-bold text-xs rounded-lg transition-colors flex items-center space-x-1.5 shadow-lg shadow-indigo-900/20"
          >
            {isAiAnalyzing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            <span>{isAiAnalyzing ? "Analyzing..." : "Run AI Analysis"}</span>
          </button>
        </div>

        {aiAnalysisResult && (
          <div className="p-4 bg-slate-950/80 rounded-lg border border-indigo-500/30 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-mono font-bold text-indigo-300 uppercase">
                AI ANALYSIS SUMMARY ({aiAnalysisResult.model_used})
              </span>
              <span className="text-xs font-mono text-emerald-400">Confidence: {aiAnalysisResult.confidence_formatted}</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed font-sans">{aiAnalysisResult.ai_explanation}</p>
          </div>
        )}
      </div>
    </div>
  );
}
