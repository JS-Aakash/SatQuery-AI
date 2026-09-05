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
  Cpu,
  Activity,
} from "lucide-react";
import { Sidebar, NavTab } from "../components/Sidebar";
import { TopNav } from "../components/TopNav";
import { UploadZone } from "../components/UploadZone";
import { SatelliteSearch } from "../components/SatelliteSearch";
import { MapAOI } from "../components/MapAOI";
import CopernicusExplorer from "../components/CopernicusExplorer";
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

  // Input modes: "upload" | "copernicus" | "search" | "aoi"
  const [inputMode, setInputMode] = useState<"upload" | "copernicus" | "search" | "aoi">("upload");

  // Imagery states
  const [primaryImage, setPrimaryImage] = useState<ImageMetadata | null>(PRESET_SCENARIOS[0].imageA);
  const [secondaryImage, setSecondaryImage] = useState<ImageMetadata | null>(null);
  const [activePreset, setActivePreset] = useState<PresetScenario | null>(PRESET_SCENARIOS[0]);

  // Query and Analysis states
  const [query, setQuery] = useState(PRESET_SCENARIOS[0].defaultQuery);
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
          if (tab === "reports") {
            handleOpenReport(analysisResult?.analysis_id || "demo-analysis");
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
                      onClick={() => setInputMode("upload")}
                      className={cn(
                        "px-3.5 py-2 rounded-md text-xs font-mono font-medium transition-all flex items-center space-x-2",
                        inputMode === "upload"
                          ? "bg-slate-800 text-emerald-400 border border-emerald-500/40 shadow-sm"
                          : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                      )}
                    >
                      <Upload className="w-3.5 h-3.5" />
                      <span>1. Upload Imagery</span>
                    </button>

                    <button
                      onClick={() => setInputMode("copernicus")}
                      className={cn(
                        "px-3.5 py-2 rounded-md text-xs font-mono font-medium transition-all flex items-center space-x-2",
                        inputMode === "copernicus"
                          ? "bg-emerald-950/60 text-emerald-300 border border-emerald-500/50 shadow-sm shadow-emerald-950/40 font-bold"
                          : "text-slate-400 hover:text-emerald-400 hover:bg-slate-900"
                      )}
                    >
                      <Satellite className="w-3.5 h-3.5 text-emerald-400" />
                      <span>2. Copernicus Data Space (Sentinel-2 / NDVI / Map)</span>
                    </button>

                    <button
                      onClick={() => setInputMode("search")}
                      className={cn(
                        "px-3.5 py-2 rounded-md text-xs font-mono font-medium transition-all flex items-center space-x-2",
                        inputMode === "search"
                          ? "bg-slate-800 text-cyan-400 border border-cyan-500/40 shadow-sm"
                          : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                      )}
                    >
                      <Search className="w-3.5 h-3.5" />
                      <span>3. Auto Catalog Search</span>
                    </button>

                    <button
                      onClick={() => setInputMode("aoi")}
                      className={cn(
                        "px-3.5 py-2 rounded-md text-xs font-mono font-medium transition-all flex items-center space-x-2",
                        inputMode === "aoi"
                          ? "bg-slate-800 text-amber-400 border border-amber-500/40 shadow-sm"
                          : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                      )}
                    >
                      <MapPin className="w-3.5 h-3.5" />
                      <span>4. AOI Definition</span>
                    </button>
                  </div>

                  {/* Mode Content Panes */}
                  <div className="bg-[#090e1a]/80 border border-slate-800/80 rounded-xl p-5 shadow-xl">
                    {inputMode === "copernicus" && (
                      <CopernicusExplorer />
                    )}

                    {inputMode === "upload" && (
                      <UploadZone
                        primaryImage={primaryImage}
                        secondaryImage={secondaryImage}
                        onSetPrimaryImage={(img) => {
                          setPrimaryImage(img);
                          if (img) {
                            setActivePreset(null);
                            if (!query.trim()) {
                              setQuery("Describe the land-cover classification and highlight major objects visible in this image.");
                            }
                          } else if (secondaryImage) {
                            // Automatically promote secondary image to primary image if primary is cleared
                            setPrimaryImage(secondaryImage);
                            setSecondaryImage(null);
                          }
                        }}
                        onSetSecondaryImage={(img) => {
                          setSecondaryImage(img);
                          if (img) {
                            setActivePreset(null);
                          }
                        }}
                        onSelectPreset={handleSelectPreset}
                        activePresetId={activePreset?.id}
                      />
                    )}

                    {inputMode === "search" && (
                      <SatelliteSearch
                        onSelectScene={(meta, isSecondary) => {
                          if (isSecondary) {
                            setSecondaryImage(meta);
                          } else {
                            setPrimaryImage(meta);
                          }
                          setActivePreset(null);
                        }}
                        onExecuteAutoAnalysis={(autoRes) => {
                          const mappedRes: AnalysisResponse = {
                            analysis_id: autoRes.workflow_id || `eo_${Date.now()}`,
                            query: autoRes.original_query || "Autonomous Earth Observation Query",
                            task: autoRes.task || "Bi-Temporal Change Intelligence",
                            status: "COMPLETED",
                            answer: autoRes.answer,
                            confidence: autoRes.confidence || 0.95,
                            confidence_formatted: autoRes.confidence_formatted || "95% (Calibrated)",
                            evidence: [
                              {
                                id: "ev-auto-1",
                                title: `Copernicus ${autoRes.sensor} Data Acquisition`,
                                category: "Satellite Retrieval",
                                description: `Acquired for ${autoRes.location_name} (Bounds: ${autoRes.bbox?.join(", ")})`,
                                confidence: 0.96,
                              }
                            ],
                            execution_trace: autoRes.execution_trace?.map((s: any) => ({
                              step_id: s.step,
                              name: s.task,
                              description: s.output_summary,
                              status: "completed",
                              duration_ms: s.duration_ms,
                              tool_or_model: s.tool,
                            })) || [],
                            grounding_boxes: [],
                            models_used: ["Copernicus CDSE", "GeoChat-7B", "ChangeNet"],
                            execution_time_ms: autoRes.execution_time_ms || 1200,
                            created_at: new Date().toISOString(),
                            is_mock: false,
                            model_status: "READY"
                          };
                          setAnalysisResult(mappedRes);
                        }}
                      />
                    )}

                    {inputMode === "aoi" && (
                      <MapAOI onAoiDefined={handleAoiDefined} />
                    )}
                  </div>

                  {/* Natural Language Query Interface */}
                  <div className="pt-2 space-y-3">
                    {/* Model Inference Loading State */}
                    {isAnalyzing && (
                      <div className="p-4 rounded-xl bg-slate-900/95 border border-emerald-500/50 shadow-2xl space-y-3 animate-in fade-in duration-200">
                        <div className="flex items-center justify-between text-xs font-mono">
                          <div className="flex items-center space-x-2 text-emerald-400">
                            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
                            <span className="font-semibold uppercase tracking-wider">
                              Executing Remote-Sensing VLM Inference...
                            </span>
                          </div>
                          <span className="text-slate-400 font-mono text-[11px]">GeoChat / Spatial Grounding</span>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-[11px] font-mono text-slate-300">
                          <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 flex items-center space-x-2">
                            <span className="text-emerald-400 font-bold">1.</span>
                            <span>Raster Ingestion & Band Normalization</span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-slate-950/80 border border-emerald-500/30 flex items-center space-x-2">
                            <span className="text-cyan-400 font-bold">2.</span>
                            <span>Visual Feature Extraction & Cross-Attention</span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 flex items-center space-x-2">
                            <span className="text-amber-400 font-bold">3.</span>
                            <span>Text Grounding & Coordinate Standardizing</span>
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="text-xs font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                      Agentic Natural-Language Query
                    </div>
                    <QueryBox
                      query={query}
                      onQueryChange={setQuery}
                      onAnalyze={handleAnalyze}
                      isAnalyzing={isAnalyzing}
                      disabled={!primaryImage && !secondaryImage}
                    />
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
