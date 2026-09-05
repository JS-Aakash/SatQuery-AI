"use client";

import React, { useState, useEffect } from "react";
import {
  Satellite,
  Compass,
  Layers,
  Sparkles,
  ArrowRight,
  Upload,
  Search,
  MapPin,
  CheckCircle2,
  AlertCircle,
  RotateCcw,
  RefreshCw,
  Cpu,
  Activity,
} from "lucide-react";
import { Sidebar, NavTab } from "../components/Sidebar";
import { TopNav } from "../components/TopNav";
import { UploadZone } from "../components/UploadZone";
import { SatelliteSearch } from "../components/SatelliteSearch";
import { MapAOI } from "../components/MapAOI";
import CopernicusExplorer from "../components/CopernicusExplorer";
import GeospatialSearch from "../components/GeospatialSearch";
import SingleImageStudio from "../components/SingleImageStudio";
import { QueryBox } from "../components/QueryBox";

import { ImageryViewer } from "../components/ImageryViewer";
import { AIAnswerPanel } from "../components/AIAnswerPanel";
import { HistoryView } from "../components/HistoryView";
import { EvaluationView } from "../components/EvaluationView";
import { SystemStatusModal } from "../components/SystemStatusModal";
import { ReportModal } from "../components/ReportModal";

import { ImageMetadata, AnalysisResponse, AnalysisRequest } from "../lib/types";
import { executeAnalysis, fetchHealth } from "../lib/api";
import {
  PRESET_SCENARIOS,
  PresetScenario,
  generateSyntheticSatelliteDataUri,
} from "../lib/presets";
import { cn } from "../lib/utils";

export default function Home() {
  // Navigation & Modal states
  const [activeTab, setActiveTab] = useState<NavTab>("home");
  const [currentWorkspace, setCurrentWorkspace] = useState("ISRO SAC Cartosat/RISAT Evaluation Set");
  const [isSystemStatusOpen, setIsSystemStatusOpen] = useState(false);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [activeReportAnalysisId, setActiveReportAnalysisId] = useState<string | null>(null);

  // Backend connection status
  const [backendOnline, setBackendOnline] = useState(false);
  const [latencyMs, setLatencyMs] = useState(28);

  // Input modes: "single" | "geospatial" | "copernicus" | "bitemporal"
  const [inputMode, setInputMode] = useState<"single" | "geospatial" | "copernicus" | "bitemporal">("single");

  // Imagery states
  const [primaryImage, setPrimaryImage] = useState<ImageMetadata | null>(null);
  const [secondaryImage, setSecondaryImage] = useState<ImageMetadata | null>(null);
  const [activePreset, setActivePreset] = useState<PresetScenario | null>(null);

  // Query and Analysis states
  const [query, setQuery] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Check backend health periodically
  useEffect(() => {
    const check = async () => {
      const start = Date.now();
      const res = await fetchHealth();
      const dur = Date.now() - start;
      setBackendOnline(res.status === "healthy");
      setLatencyMs(dur);
    };
    check();
    const timer = setInterval(check, 10000);
    return () => clearInterval(timer);
  }, []);

  // Handle preset selection
  const handleSelectPreset = (preset: PresetScenario) => {
    setActivePreset(preset);
    setPrimaryImage(preset.imageA);
    setSecondaryImage(preset.imageB || null);
    setQuery(preset.defaultQuery);
    setAnalysisResult(null);
    setErrorMessage(null);
  };

  // Handle AOI selection from map
  const handleAoiDefined = (coords: [number, number][], areaKm2: number, locName: string) => {
    // Generate synthetic metadata for this AOI
    const customMeta: ImageMetadata = {
      id: `img-aoi-${Date.now()}`,
      filename: `AOI_${locName.replace(/[^a-zA-Z0-9]/g, "_")}.tif`,
      file_size_bytes: Math.round(areaKm2 * 1024 * 1024 * 0.15),
      file_size_formatted: `${(areaKm2 * 0.15).toFixed(1)} MB`,
      format: "GeoTIFF",
      dimensions: [1024, 1024],
      bands: 12,
      band_names: ["B02 Blue", "B03 Green", "B04 Red", "B08 NIR", "B11 SWIR"],
      crs: "EPSG:4326 (WGS 84)",
      bounding_box: [coords[0][0], coords[0][1], coords[2][0], coords[2][1]],
      modality: "Multispectral",
      sensor: "Sentinel-2",
      acquisition_date: new Date().toISOString(),
      is_valid: true,
      validation_notes: [`AOI mapped (${areaKm2} km²)`],
    };
    setPrimaryImage(customMeta);
    setSecondaryImage(null);
    setActivePreset(null);
  };

  // Run Agentic Analysis
  const handleAnalyze = async () => {
    if (!query.trim()) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    const request: AnalysisRequest = {
      query: query.trim(),
      image_metadata: primaryImage || undefined,
      secondary_image_metadata: secondaryImage || undefined,
      input_mode: inputMode,
      image_data_uri: imageAUrl || undefined,
    };

    try {
      const res = await executeAnalysis(request);
      setAnalysisResult(res);
      setActiveReportAnalysisId(res.analysis_id);
    } catch (err: any) {
      setErrorMessage("Agent analysis execution encountered an issue. Using cached telemetry pipeline.");
      // Graceful fallback for UI stability
      const fallbackReq: AnalysisRequest = {
        query: query.trim(),
        preferred_task: secondaryImage ? "Bi-Temporal Change Analysis" : "Visual Question Answering",
      };
      const fallbackRes = await executeAnalysis(fallbackReq);
      setAnalysisResult(fallbackRes);
      setActiveReportAnalysisId(fallbackRes.analysis_id);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleOpenReport = (analysisId?: string) => {
    if (analysisId) {
      setActiveReportAnalysisId(analysisId);
    }
    setIsReportModalOpen(true);
  };

  const handleResetQuery = () => {
    setAnalysisResult(null);
  };

  // Determine current imagery render URLs
  const sampleTypeA = activePreset ? activePreset.sampleTypeA : (primaryImage?.modality === "SAR" ? "sar" : "optical");
  const sampleTypeB = activePreset?.sampleTypeB || (secondaryImage?.modality === "SAR" ? "sar" : "temporal_t2");

  const imageAUrl = primaryImage?.preview_url || generateSyntheticSatelliteDataUri(primaryImage?.filename || "Primary Scene", sampleTypeA);
  const imageBUrl = secondaryImage?.preview_url || (secondaryImage ? generateSyntheticSatelliteDataUri(secondaryImage.filename, sampleTypeB) : undefined);

  return (
    <div className="flex min-h-screen bg-[#070b14] text-slate-100 font-sans antialiased">
      {/* Persistent Left Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={(tab) => {
          setActiveTab(tab);
          if (tab === "reports" && analysisResult?.analysis_id) {
            handleOpenReport(analysisResult.analysis_id);
          }
        }}
        onOpenSystemStatus={() => setIsSystemStatusOpen(true)}
        backendOnline={backendOnline}
      />

      {/* Main Mission Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <TopNav
          currentWorkspace={currentWorkspace}
          onSelectWorkspace={setCurrentWorkspace}
          onOpenSystemStatus={() => setIsSystemStatusOpen(true)}
          backendOnline={backendOnline}
          latencyMs={latencyMs}
        />

        <main className="flex-1 p-6 overflow-y-auto">
          {/* TAB 1: NEW ANALYSIS (HOME) */}
          {activeTab === "home" && (
            <div className="space-y-6 max-w-7xl mx-auto">
              {/* Hero Section */}
              <div className="space-y-2 text-left pb-2">
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 tracking-wider uppercase">
                    ISRO PS ID: 26167
                  </span>
                  <span className="text-[10px] font-mono text-slate-500">
                    MULTIMODAL REMOTE SENSING AI
                  </span>
                </div>
                <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-100 flex items-center gap-3">
                  <span>SATQUERY AI</span>
                  <span className="text-xs font-mono font-normal px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-400">
                    Agentic Mission Control
                  </span>
                </h1>
                <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
                  <strong className="text-emerald-400 font-medium">"Ask your satellite imagery anything."</strong>{" "}
                  Analyze optical, multispectral, synthetic aperture radar (SAR), and multitemporal remote sensing imagery
                  through natural-language queries. Powered by query-driven agentic tool orchestration and spatial evidence grounding.
                </p>
              </div>

              {/* View 1: If an analysis has completed, show the RESULT VIEW */}
              {analysisResult ? (
                <div className="space-y-4 animate-in fade-in duration-200">
                  {/* Result View Top Bar */}
                  <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-xs font-mono">
                    <div className="flex items-center space-x-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 radar-live-dot" />
                      <span className="text-slate-300 font-semibold">ANALYSIS RESULT ACTIVE</span>
                      <span className="text-slate-600">|</span>
                      <span className="text-slate-400">Query: "{analysisResult.query}"</span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <button
                        onClick={handleResetQuery}
                        className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors flex items-center space-x-1.5"
                      >
                        <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
                        <span>Modify Query & Inputs</span>
                      </button>
                    </div>
                  </div>

                  {/* Dual Grid: Left Imagery Viewer + Right AI Answer Panel */}
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
                    {/* Left: Large Imagery Viewer (7 cols) */}
                    <div className="lg:col-span-7 h-full">
                      <ImageryViewer
                        imageAUrl={imageAUrl}
                        imageBUrl={imageBUrl}
                        imageAName={primaryImage?.filename || "Observation T1"}
                        imageBName={secondaryImage?.filename || "Observation T2 / SAR"}
                        groundingBoxes={analysisResult.grounding_boxes}
                        changeMap={analysisResult.change_map}
                        hasTemporalComparison={Boolean(secondaryImage)}
                        selectedBoxId={selectedBoxId}
                        onSelectBox={setSelectedBoxId}
                      />
                    </div>

                    {/* Right: AI Answer & Execution Trace Panel (5 cols) */}
                    <div className="lg:col-span-5 h-full">
                      <AIAnswerPanel
                        analysis={analysisResult}
                        onOpenReportModal={() => handleOpenReport(analysisResult.analysis_id)}
                        onResetAnalysis={handleResetQuery}
                        selectedBoxId={selectedBoxId}
                        onSelectBox={setSelectedBoxId}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                /* View 2: Analysis Setup Workspace (3 Input Modes + Query Interface) */
                <div className="space-y-6">
                  {/* Mode Selector Tabs */}
                  <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-2">
                    <button
                      onClick={() => setInputMode("single")}
                      className={cn(
                        "px-3.5 py-2 rounded-md text-xs font-mono font-medium transition-all flex items-center space-x-2",
                        inputMode === "single"
                          ? "bg-emerald-950/70 text-emerald-300 border border-emerald-500/50 shadow-sm shadow-emerald-950/40 font-bold"
                          : "text-slate-400 hover:text-emerald-400 hover:bg-slate-900"
                      )}
                    >
                      <Compass className="w-3.5 h-3.5 text-emerald-400" />
                      <span>1. Single-Image Studio (Optical / SAR / Pairs)</span>
                    </button>

                    <button
                      onClick={() => setInputMode("geospatial")}
                      className={cn(
                        "px-3.5 py-2 rounded-md text-xs font-mono font-medium transition-all flex items-center space-x-2",
                        inputMode === "geospatial"
                          ? "bg-cyan-950/70 text-cyan-300 border border-cyan-500/50 shadow-sm shadow-cyan-950/40 font-bold"
                          : "text-slate-400 hover:text-cyan-400 hover:bg-slate-900"
                      )}
                    >
                      <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                      <span>2. Natural-Language Geospatial Search</span>
                    </button>

                    <button
                      onClick={() => setInputMode("copernicus")}
                      className={cn(
                        "px-3.5 py-2 rounded-md text-xs font-mono font-medium transition-all flex items-center space-x-2",
                        inputMode === "copernicus"
                          ? "bg-blue-950/60 text-blue-300 border border-blue-500/50 shadow-sm shadow-blue-950/40 font-bold"
                          : "text-slate-400 hover:text-blue-400 hover:bg-slate-900"
                      )}
                    >
                      <Satellite className="w-3.5 h-3.5 text-blue-400" />
                      <span>3. Copernicus Data Space & Spectral Analysis</span>
                    </button>

                    <button
                      onClick={() => setInputMode("bitemporal")}
                      className={cn(
                        "px-3.5 py-2 rounded-md text-xs font-mono font-medium transition-all flex items-center space-x-2",
                        inputMode === "bitemporal"
                          ? "bg-purple-950/60 text-purple-300 border border-purple-500/50 shadow-sm shadow-purple-950/40 font-bold"
                          : "text-slate-400 hover:text-purple-400 hover:bg-slate-900"
                      )}
                    >
                      <Layers className="w-3.5 h-3.5 text-purple-400" />
                      <span>4. Bi-Temporal Change Detection</span>
                    </button>
                  </div>

                  {/* Mode Content Panes */}
                  <div className="bg-[#090e1a]/80 border border-slate-800/80 rounded-xl p-5 shadow-xl">
                    {inputMode === "single" && (
                      <SingleImageStudio />
                    )}

                    {inputMode === "geospatial" && (
                      <GeospatialSearch />
                    )}

                    {inputMode === "copernicus" && (
                      <CopernicusExplorer />
                    )}

                    {inputMode === "bitemporal" && (
                      <div className="space-y-6">
                        <UploadZone
                          primaryImage={primaryImage}
                          secondaryImage={secondaryImage}
                          onSetPrimaryImage={(img) => setPrimaryImage(img)}
                          onSetSecondaryImage={(img) => setSecondaryImage(img)}
                        />

                        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-mono text-purple-400 font-semibold uppercase flex items-center gap-1.5">
                              <Sparkles className="w-4 h-4 text-purple-400" />
                              Bi-Temporal Change Query & Intelligence
                            </span>
                            {primaryImage && secondaryImage ? (
                              <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                                <CheckCircle2 className="w-3 h-3" /> Both Observations Ready
                              </span>
                            ) : (
                              <span className="text-[10px] font-mono text-slate-500">
                                Upload Date 1 & Date 2 rasters to enable
                              </span>
                            )}
                          </div>

                          <div className="flex gap-2">
                            <input
                              type="text"
                              value={query || "Analyze bi-temporal land-cover changes, vegetation loss, and urban growth."}
                              onChange={(e) => setQuery(e.target.value)}
                              onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
                              placeholder="e.g. Quantify urban expansion, Detect vegetation loss between Date 1 and Date 2..."
                              className="flex-1 px-3.5 py-2 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-100 text-sm focus:outline-none focus:border-purple-500 font-mono"
                            />
                            <button
                              onClick={handleAnalyze}
                              disabled={isAnalyzing || !primaryImage || !secondaryImage}
                              className="px-5 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 disabled:bg-slate-800 text-white font-semibold text-xs uppercase font-mono tracking-wider transition-colors flex items-center gap-1.5 shadow-md shadow-purple-950/40"
                            >
                              {isAnalyzing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                              Execute Change Analysis
                            </button>
                          </div>

                          <div className="flex flex-wrap gap-1.5 pt-1">
                            {[
                              "Analyze urban expansion and new construction",
                              "Quantify vegetation loss and agricultural canopy decline",
                              "Detect surface water body depletion or flood inundation",
                              "Identify persistent bare land candidates across both epochs",
                            ].map((chip, idx) => (
                              <button
                                key={idx}
                                onClick={() => {
                                  setQuery(chip);
                                  if (primaryImage && secondaryImage) {
                                    handleAnalyze();
                                  }
                                }}
                                className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[11px] text-slate-300 font-mono transition-colors"
                              >
                                {chip}
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: WORKSPACE (PERSISTENT IMAGERY WORKBENCH) */}
          {activeTab === "workspace" && (
            <div className="space-y-6 max-w-7xl mx-auto">
              <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold uppercase tracking-wider font-mono text-slate-100">
                    Active Mission Workspace: {currentWorkspace}
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Multi-sensor geospatial workbench with real-time inspection, band synthesis, and co-registration telemetry.
                  </p>
                </div>
                <button
                  onClick={() => setActiveTab("home")}
                  className="px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-semibold text-xs transition-colors"
                >
                  Analyze in Studio
                </button>
              </div>

              {primaryImage && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                  <div className="lg:col-span-8">
                    <ImageryViewer
                      imageAUrl={imageAUrl}
                      imageBUrl={imageBUrl}
                      imageAName={primaryImage.filename}
                      imageBName={secondaryImage?.filename}
                      hasTemporalComparison={Boolean(secondaryImage)}
                    />
                  </div>
                  <div className="lg:col-span-4 space-y-4">
                    <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3 text-xs font-mono">
                      <div className="text-slate-400 uppercase font-semibold">
                        Raster Metadata Inspection
                      </div>
                      <div className="space-y-1.5 text-slate-300">
                        <div><span className="text-slate-500">File:</span> {primaryImage.filename}</div>
                        <div><span className="text-slate-500">Sensor:</span> {primaryImage.sensor}</div>
                        <div><span className="text-slate-500">Modality:</span> {primaryImage.modality}</div>
                        <div><span className="text-slate-500">Format:</span> {primaryImage.format}</div>
                        <div><span className="text-slate-500">Bands:</span> {primaryImage.bands} Channels</div>
                        <div><span className="text-slate-500">CRS:</span> {primaryImage.crs}</div>
                        <div><span className="text-slate-500">Acquisition:</span> {primaryImage.acquisition_date}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: ANALYSIS HISTORY */}
          {activeTab === "history" && (
            <HistoryView
              onOpenAnalysis={(id) => {
                setActiveReportAnalysisId(id);
                setActiveTab("home");
              }}
              onOpenReportModal={handleOpenReport}
            />
          )}

          {/* TAB 4: SAVED REPORTS (TRIGGER REPORT MODAL) */}
          {activeTab === "reports" && (
            <div className="space-y-6 max-w-7xl mx-auto">
              <div className="p-8 text-center rounded-xl bg-slate-900/50 border border-slate-800 space-y-3">
                <Satellite className="w-8 h-8 text-emerald-400 mx-auto" />
                <h3 className="text-sm font-semibold text-slate-200">
                  Mission Reports & Compliance Documentation
                </h3>
                <p className="text-xs text-slate-400 max-w-lg mx-auto">
                  Auditable mission reports in Markdown and JSON formats including confidence breakdowns,
                  calibrated evidence items, and complete specialist tool execution traces.
                </p>
                <button
                  onClick={() => handleOpenReport("sample-report")}
                  className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-semibold text-xs font-mono transition-colors"
                >
                  Generate Current Mission Report
                </button>
              </div>
            </div>
          )}

          {/* TAB 5: EVALUATION BENCHMARK DASHBOARD */}
          {activeTab === "evaluation" && <EvaluationView />}

          {/* TAB 6: SETTINGS */}
          {activeTab === "settings" && (
            <div className="max-w-4xl mx-auto space-y-6">
              <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <h2 className="text-sm font-semibold uppercase tracking-wider font-mono text-slate-100">
                  Mission Platform Configuration & Preferences
                </h2>
                <p className="text-xs text-slate-400">
                  Configure API connections, satellite credentials, and specialist model registry endpoints.
                </p>
              </div>

              <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-4 text-xs font-mono">
                <div className="space-y-1.5">
                  <label className="text-slate-400 font-semibold">BACKEND API ENDPOINT</label>
                  <input
                    type="text"
                    defaultValue="http://127.0.0.1:8000/api"
                    className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-slate-200 text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-slate-400 font-semibold">COPERNICUS STAC CATALOG API</label>
                  <input
                    type="text"
                    defaultValue="https://catalogue.dataspace.copernicus.eu/stac"
                    className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-slate-200 text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <div className="pt-2 flex items-center justify-between border-t border-slate-800">
                  <span className="text-slate-400">ISRO SAC Evaluation Annotation Keys:</span>
                  <span className="text-amber-400">Restricted (Undisclosed)</span>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* System Diagnostics Modal */}
      <SystemStatusModal
        isOpen={isSystemStatusOpen}
        onClose={() => setIsSystemStatusOpen(false)}
      />

      {/* Mission Intelligence Report Modal */}
      <ReportModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        analysisId={activeReportAnalysisId}
      />
    </div>
  );
}
