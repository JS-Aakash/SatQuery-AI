"use client";
import React, { useState, useEffect, useRef } from "react";
import {
  UploadCloud,
  FileCheck,
  CheckCircle2,
  AlertCircle,
  Layers,
  Sparkles,
  RefreshCw,
  X,
  Sliders,
  Eye,
  MapPin,
  Compass,
  Maximize2,
  Minimize2,
  Info,
  ShieldAlert,
  Search,
  Activity,
  ChevronRight,
  Database,
  ArrowRight,
  Wand2,
  Download,
} from "lucide-react";
import {
  ImageMetadata,
  GroundedRegionData,
  SingleImageAnalyzeResponse,
} from "../lib/types";
import {
  uploadSingleGeoTIFF,
  buildMultispectralGeoTIFF,
  analyzeSingleGeospatialImage,
  fetchCompositePreview,
  getDownloadUrl,
} from "../lib/api";
import { cn } from "../lib/utils";
import dynamic from "next/dynamic";
import { ImageryViewer } from "./ImageryViewer";

// Dynamically import Leaflet Map component to avoid SSR issues
const SatelliteMap = dynamic(() => import("./SatelliteMap"), {
  ssr: false,
  loading: () => (
    <div className="h-[400px] w-full rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-center text-slate-500 font-mono text-xs">
      <RefreshCw className="w-5 h-5 animate-spin text-emerald-400 mr-2" />
      INITIALIZING GEOSPATIAL MAP VIEWPORT...
    </div>
  ),
});

interface SingleImageStudioProps {
  initialMetadata?: ImageMetadata | null;
  onAnalysisCompleted?: (res: SingleImageAnalyzeResponse) => void;
}

export const SingleImageStudio: React.FC<SingleImageStudioProps> = ({
  initialMetadata = null,
  onAnalysisCompleted,
}) => {
  const [imageMeta, setImageMeta] = useState<ImageMetadata | null>(initialMetadata);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [query, setQuery] = useState("Find potentially vacant land");
  const [analysisResult, setAnalysisResult] = useState<SingleImageAnalyzeResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [selectedComposite, setSelectedComposite] = useState<string>("true_color");
  const [isBuilderOpen, setIsBuilderOpen] = useState(false);
  const [isBuildingMultispectral, setIsBuildingMultispectral] = useState(false);
  const [builtMeta, setBuiltMeta] = useState<ImageMetadata | null>(null);

  useEffect(() => {
    if (initialMetadata) {
      setImageMeta(initialMetadata);
    }
  }, [initialMetadata]);

  // Multispectral builder single-band file states
  const [builderFiles, setBuilderFiles] = useState<{
    b02?: File;
    b03?: File;
    b04?: File;
    b08?: File;
    b11?: File;
    b12?: File;
  }>({});

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setErrorMessage(null);
    setAnalysisResult(null);

    try {
      const res = await uploadSingleGeoTIFF(file);
      let meta = res.metadata;
      if (!meta.preview_url && meta.id) {
        const isSar = meta.modality === "SAR" || (meta.raster_type && meta.raster_type.includes("SAR"));
        const defaultComp = isSar ? "sar_db" : "true_color";
        const autoPreview = await fetchCompositePreview(meta.id, defaultComp);
        if (autoPreview) {
          meta = { ...meta, preview_url: autoPreview };
        }
      }
      setImageMeta(meta);
      setIsUploading(false);
    } catch (err: any) {
      setIsUploading(false);
      setErrorMessage(err.message || "Failed to inspect GeoTIFF raster.");
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async (overrideQuery?: string) => {
    const q = overrideQuery || query;
    if (!q.trim() || !imageMeta) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const res = await analyzeSingleGeospatialImage({
        imageId: imageMeta.id,
        query: q.trim(),
        task: "auto",
      });
      setAnalysisResult(res);
      setIsAnalyzing(false);
      if (onAnalysisCompleted) {
        onAnalysisCompleted(res);
      }
    } catch (err: any) {
      setIsAnalyzing(false);
      setErrorMessage(err.message || "Analysis execution encountered an issue.");
    }
  };

  const handleSwitchComposite = async (type: "true_color" | "false_color_nir" | "sar_db") => {
    if (!imageMeta) return;
    setSelectedComposite(type);
    const newUrl = await fetchCompositePreview(imageMeta.id, type);
    if (newUrl) {
      setImageMeta({ ...imageMeta, preview_url: newUrl });
    }
  };

  const handleBuildMultispectral = async () => {
    setIsBuildingMultispectral(true);
    setErrorMessage(null);
    try {
      const res = await buildMultispectralGeoTIFF(builderFiles);
      setBuiltMeta(res.metadata);
      setImageMeta(res.metadata);
      setIsBuildingMultispectral(false);
    } catch (err: any) {
      setIsBuildingMultispectral(false);
      setErrorMessage(err.message || "Failed to assemble multispectral GeoTIFF.");
    }
  };

  // Prepare grounding bounding boxes for visual overlay on image canvas
  const viewerGroundingBoxes = (analysisResult?.bounding_boxes && analysisResult.bounding_boxes.length > 0)
    ? analysisResult.bounding_boxes.map((b) => ({
        id: b.id,
        label: b.label,
        box: b.box as [number, number, number, number],
        confidence: b.confidence,
        color: b.color || "#10b981",
      }))
    : (analysisResult?.grounded_regions && analysisResult.grounded_regions.length > 0)
    ? analysisResult.grounded_regions.map((r) => ({
        id: r.id,
        label: r.label,
        box: (r.pixel_bbox && r.pixel_bbox.length === 4 ? r.pixel_bbox : [20, 20, 60, 60]) as [number, number, number, number],
        confidence: r.confidence,
        color: r.color || "#10b981",
      }))
    : [];

  // Prepare map polygon data for Leaflet SatelliteMap
  const mapPolygons = analysisResult?.grounded_regions?.map((r) => ({
    id: r.id,
    label: `${r.label} (${r.area_ha.toFixed(2)} ha)`,
    coordinates: r.polygon.coordinates[0],
    color: r.color || "#10b981",
  })) || [];

  return (
    <div className="space-y-6">
      {/* 1. Header & Mode Switcher */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 p-4 rounded-xl bg-slate-900/90 border border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              <Compass className="w-4 h-4" />
            </span>
            <h2 className="text-sm font-semibold text-slate-100 uppercase tracking-wider font-mono">
              Single-Image Geospatial Multimodal Studio
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Analyze one optical multispectral or SAR GeoTIFF with physics-grounded VQA, Captioning, and Georeferenced Grounding.
          </p>
        </div>

        <button
          onClick={() => setIsBuilderOpen(true)}
          className="px-3 py-1.5 rounded-lg bg-cyan-950/40 hover:bg-cyan-900/50 border border-cyan-800/60 text-cyan-300 text-xs font-mono flex items-center gap-1.5 transition-colors"
        >
          <Wand2 className="w-3.5 h-3.5 text-cyan-400" />
          Create Multispectral GeoTIFF
        </button>
      </div>

      {/* 2. Upload One Geospatial Image Zone */}
      {!imageMeta ? (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-800 hover:border-emerald-500/60 bg-slate-900/40 hover:bg-slate-900/70 rounded-2xl p-10 text-center cursor-pointer transition-all space-y-4"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".tif,.tiff"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
              }
            }}
            className="hidden"
          />

          <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mx-auto text-emerald-400 shadow-inner">
            <UploadCloud className="w-7 h-7" />
          </div>

          <div>
            <h3 className="text-base font-semibold text-slate-200">
              Upload one geospatial image
            </h3>
            <p className="text-xs text-slate-400 mt-1 font-mono">
              Supported: Optical RGB GeoTIFF, Multispectral GeoTIFF, Single-band optical, or Sentinel-1 SAR (.tif, .tiff)
            </p>
          </div>

          <div className="flex items-center justify-center gap-2 pt-2 text-[11px] text-slate-500 font-mono">
            <span className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700">Sentinel-2 Multispectral</span>
            <span className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700">Sentinel-1 SAR (VV / VH)</span>
            <span className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700">Optical True Color</span>
          </div>
        </div>
      ) : (
        /* 3. Ingested Raster Metadata HUD & Interactive Imagery Viewer */
        <div className="space-y-5">
          {/* Top Row: Telemetry HUD + Channel Composite Switcher + Download */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            {/* Left: Telemetry HUD (7 cols) */}
            <div className="lg:col-span-7 p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs font-mono font-semibold text-slate-200 uppercase">
                    Raster Telemetry HUD
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {imageMeta.id && (
                    <a
                      href={getDownloadUrl(imageMeta.id)}
                      download={imageMeta.filename || "geotiff.tif"}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300 border border-slate-700 text-[11px] font-mono flex items-center gap-1 transition-colors"
                      title="Download full multi-band GeoTIFF"
                    >
                      <Download className="w-3.5 h-3.5" /> Download GeoTIFF
                    </a>
                  )}
                  <button
                    onClick={() => {
                      setImageMeta(null);
                      setAnalysisResult(null);
                    }}
                    className="text-slate-500 hover:text-slate-300 text-xs font-mono flex items-center gap-1"
                  >
                    <X className="w-3.5 h-3.5" /> Eject
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
                  <span className="text-slate-500 text-[10px] block uppercase">Type</span>
                  <span className="text-emerald-400 font-bold text-[11px] truncate block">
                    {imageMeta.raster_type || imageMeta.modality}
                  </span>
                </div>
                <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
                  <span className="text-slate-500 text-[10px] block uppercase">Sensor</span>
                  <span className="text-cyan-400 font-bold text-[11px] truncate block">
                    {imageMeta.sensor}
                  </span>
                </div>
                <div className="p-2 rounded bg-slate-800/40 border border-slate-700/40">
                  <span className="text-slate-500 text-[10px] block uppercase">Bands / GSD</span>
                  <span className="text-slate-200 font-semibold">{imageMeta.bands} bands · {imageMeta.resolution_m || 10}m</span>
                </div>
                <div className="p-2 rounded bg-slate-800/40 border border-slate-700/40">
                  <span className="text-slate-500 text-[10px] block uppercase">Size / CRS</span>
                  <span className="text-slate-200 font-semibold truncate block">
                    {imageMeta.dimensions[0]}×{imageMeta.dimensions[1]} · {imageMeta.crs ? imageMeta.crs.split(":").pop() : "WGS84"}
                  </span>
                </div>
              </div>

              {/* Band Descriptions & Spectral Biophysical Stats */}
              <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                {imageMeta.band_names && imageMeta.band_names.length > 0 && (
                  <div className="flex flex-wrap gap-1 items-center">
                    <span className="text-slate-500 text-[10px] uppercase font-mono mr-1">Bands:</span>
                    {imageMeta.band_names.map((b, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 rounded bg-slate-800 text-[10px] text-slate-300 border border-slate-700 font-mono"
                      >
                        {b}
                      </span>
                    ))}
                  </div>
                )}

                {imageMeta.spectral_indices && (
                  <div className="flex items-center gap-2 text-[10px] font-mono">
                    {imageMeta.spectral_indices.ndvi && (
                      <span className="px-2 py-0.5 rounded bg-emerald-950/40 border border-emerald-800/50 text-emerald-300">
                        NDVI: <strong>{imageMeta.spectral_indices.ndvi.mean}</strong>
                      </span>
                    )}
                    {imageMeta.spectral_indices.ndwi && (
                      <span className="px-2 py-0.5 rounded bg-cyan-950/40 border border-cyan-800/50 text-cyan-300">
                        NDWI: <strong>{imageMeta.spectral_indices.ndwi.mean}</strong>
                      </span>
                    )}
                    {imageMeta.spectral_indices.ndbi && (
                      <span className="px-2 py-0.5 rounded bg-amber-950/40 border border-amber-800/50 text-amber-300">
                        NDBI: <strong>{imageMeta.spectral_indices.ndbi.mean}</strong>
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Right: Channel Composite Controls (5 cols) */}
            <div className="lg:col-span-5 p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-mono font-semibold text-slate-200 uppercase flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  Channel Visualization
                </span>
                <span className="text-[10px] font-mono text-slate-500">
                  {imageMeta.raster_type?.includes("SAR") ? "Radar Calibrated" : "Optical Synthesis"}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 py-2">
                <button
                  onClick={() => handleSwitchComposite("true_color")}
                  className={cn(
                    "p-2 rounded-lg text-xs font-mono text-center border transition-all",
                    selectedComposite === "true_color"
                      ? "bg-slate-800 border-emerald-500 text-emerald-300 font-bold"
                      : "bg-slate-800/40 hover:bg-slate-800 border-slate-700 text-slate-400"
                  )}
                >
                  <span className="block font-semibold">True Color</span>
                  <span className="text-[9px] text-slate-500">RGB B04/03/02</span>
                </button>

                <button
                  onClick={() => handleSwitchComposite("false_color_nir")}
                  className={cn(
                    "p-2 rounded-lg text-xs font-mono text-center border transition-all",
                    selectedComposite === "false_color_nir"
                      ? "bg-slate-800 border-cyan-500 text-cyan-300 font-bold"
                      : "bg-slate-800/40 hover:bg-slate-800 border-slate-700 text-slate-400"
                  )}
                >
                  <span className="block font-semibold">False Color</span>
                  <span className="text-[9px] text-slate-500">NIR B08/04/03</span>
                </button>

                <button
                  onClick={() => handleSwitchComposite("sar_db")}
                  className={cn(
                    "p-2 rounded-lg text-xs font-mono text-center border transition-all",
                    selectedComposite === "sar_db"
                      ? "bg-slate-800 border-amber-500 text-amber-300 font-bold"
                      : "bg-slate-800/40 hover:bg-slate-800 border-slate-700 text-slate-400"
                  )}
                >
                  <span className="block font-semibold">SAR dB</span>
                  <span className="text-[9px] text-slate-500">VV / VH Backscatter</span>
                </button>
              </div>

              <div className="text-[10px] text-slate-400 font-mono flex items-center justify-between">
                <span>File: {imageMeta.filename}</span>
                <span>{imageMeta.file_size_formatted}</span>
              </div>
            </div>
          </div>

          {/* Main Interactive Viewer & AI Grounding Workspace */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
            {/* Left: Large Imagery Viewer with Bounding Box Highlighting Overlay (7 cols) */}
            <div className="lg:col-span-7 space-y-3">
              <ImageryViewer
                imageAUrl={imageMeta.preview_url || ""}
                imageAName={imageMeta.filename}
                groundingBoxes={viewerGroundingBoxes}
                selectedBoxId={selectedRegionId}
                onSelectBox={setSelectedRegionId}
              />
            </div>

            {/* Right: Question Box, AI Answer & Grounded Candidates (5 cols) */}
            <div className="lg:col-span-5 space-y-4">
            {/* Question Box */}
            <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
              <label className="text-xs font-mono uppercase tracking-wider text-slate-400 block">
                Ask a question or request region grounding
              </label>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
                  placeholder="e.g. Find potentially vacant land, Show vegetation, Where are built-up areas?"
                  className="flex-1 px-3.5 py-2 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-100 text-sm focus:outline-none focus:border-emerald-500 font-mono"
                />
                <button
                  onClick={() => handleAnalyze()}
                  disabled={isAnalyzing || !query.trim()}
                  className="px-5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 text-white font-semibold text-xs uppercase font-mono tracking-wider transition-colors flex items-center gap-1.5 shadow-md shadow-emerald-950/40"
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Analyze
                    </>
                  )}
                </button>
              </div>

              {/* Example Suggestion Chips */}
              <div className="flex flex-wrap gap-1.5 pt-1">
                {[
                  "Find potentially vacant land",
                  "Describe this image",
                  "Is this area mostly urban or agricultural?",
                  "Where are the built-up regions?",
                  "Find water bodies",
                  "Highlight regions with strong SAR backscatter",
                ].map((chip, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setQuery(chip);
                      handleAnalyze(chip);
                    }}
                    className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[11px] text-slate-300 font-mono transition-colors"
                  >
                    {chip}
                  </button>
                ))}
              </div>
            </div>

            {/* Analysis Result Card (if ready) */}
            {analysisResult && (
              <div className="p-4 rounded-xl bg-slate-900/90 border border-emerald-500/30 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-semibold text-emerald-400 uppercase flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" />
                    {analysisResult.task}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 text-[10px] font-mono border border-emerald-500/30">
                    Confidence: {analysisResult.confidence_formatted}
                  </span>
                </div>

                <p className="text-sm text-slate-200 leading-relaxed">
                  {analysisResult.answer}
                </p>

                {/* Legal Disclaimer */}
                {analysisResult.disclaimer && (
                  <div className="p-2.5 rounded bg-amber-950/30 border border-amber-800/40 text-[11px] text-amber-300/90 flex items-start gap-2">
                    <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    <span>{analysisResult.disclaimer}</span>
                  </div>
                )}

                {/* Grounded Candidate Regions */}
                {analysisResult.grounded_regions && analysisResult.grounded_regions.length > 0 && (
                  <div className="space-y-2 pt-2 border-t border-slate-800">
                    <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block">
                      Delineated Candidate Regions ({analysisResult.grounded_regions.length})
                    </span>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {analysisResult.grounded_regions.map((r) => {
                        const isSelected = selectedRegionId === r.id;
                        return (
                          <div
                            key={r.id}
                            onClick={() => setSelectedRegionId(r.id)}
                            className={cn(
                              "p-3 rounded-lg border text-left cursor-pointer transition-all space-y-1.5",
                              isSelected
                                ? "bg-slate-800/90 border-emerald-500 shadow-md shadow-emerald-950/30"
                                : "bg-slate-800/40 hover:bg-slate-800/70 border-slate-700/60"
                            )}
                          >
                            <div className="flex items-center justify-between text-xs">
                              <span className="font-semibold text-slate-200 flex items-center gap-1">
                                <span
                                  className="w-2.5 h-2.5 rounded-full"
                                  style={{ backgroundColor: r.color }}
                                />
                                {r.label}
                              </span>
                              <span className="text-emerald-400 font-mono text-[10px]">
                                {(r.confidence * 100).toFixed(0)}%
                              </span>
                            </div>

                            <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                              <span>Area: <strong className="text-slate-200">{r.area_ha.toFixed(2)} ha</strong></span>
                              <span>Centroid: {r.centroid[0].toFixed(3)}°, {r.centroid[1].toFixed(3)}°</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Interactive Leaflet Map Rendering Georeferenced Image + Grounded Polygons */}
            <div className="rounded-xl overflow-hidden border border-slate-800">
              <div className="p-2.5 bg-slate-900 border-b border-slate-800 flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <MapPin className="w-3.5 h-3.5 text-emerald-400" />
                  Georeferenced Map Viewport
                </span>
                <span className="text-slate-500 text-[10px]">
                  {mapPolygons.length} vector polygons mapped
                </span>
              </div>

              <SatelliteMap
                center={
                  imageMeta.bounds_wgs84
                    ? [(imageMeta.bounds_wgs84[1] + imageMeta.bounds_wgs84[3]) / 2, (imageMeta.bounds_wgs84[0] + imageMeta.bounds_wgs84[2]) / 2]
                    : [11.275, 77.583]
                }
                zoom={14}
                polygons={mapPolygons}
                selectedPolygonId={selectedRegionId || undefined}
                onSelectPolygon={(poly) => setSelectedRegionId(poly.id)}
              />
            </div>
          </div>
        </div>
      </div>
    )}

      {/* 4. Multispectral Builder Modal */}
      {isBuilderOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="max-w-lg w-full rounded-2xl bg-slate-900 border border-slate-700 p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Wand2 className="w-5 h-5 text-cyan-400" />
                <h3 className="text-sm font-semibold text-slate-100 uppercase font-mono">
                  Create Multispectral GeoTIFF
                </h3>
              </div>
              <button
                onClick={() => setIsBuilderOpen(false)}
                className="text-slate-500 hover:text-slate-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Upload individual Sentinel-2 single-band TIFFs (B02, B03, B04, B08, B11, B12). The builder will resample 20m SWIR bands to match the 10m grid and package them into a single multi-band GeoTIFF.
            </p>

            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              {(["B02", "B03", "B04", "B08", "B11", "B12"] as const).map((b) => {
                const key = b.toLowerCase() as keyof typeof builderFiles;
                const isSelected = !!builderFiles[key];
                return (
                  <div
                    key={b}
                    className={cn(
                      "p-2.5 rounded-lg border flex items-center justify-between",
                      isSelected
                        ? "bg-slate-800 border-cyan-500/60 text-cyan-300"
                        : "bg-slate-900/60 border-slate-800 text-slate-400"
                    )}
                  >
                    <span>{b} Band</span>
                    <label className="cursor-pointer px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] text-slate-200">
                      {isSelected ? "Change" : "Browse"}
                      <input
                        type="file"
                        accept=".tif,.tiff"
                        className="hidden"
                        onChange={(e) => {
                          if (e.target.files && e.target.files.length > 0) {
                            setBuilderFiles((prev) => ({
                              ...prev,
                              [key]: e.target.files![0],
                            }));
                          }
                        }}
                      />
                    </label>
                  </div>
                );
              })}
            </div>

            {builtMeta && (
              <div className="p-3.5 rounded-xl bg-emerald-950/60 border border-emerald-500/50 space-y-2.5">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-emerald-300 font-semibold flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    Multispectral GeoTIFF Ready!
                  </span>
                  <span className="text-emerald-400/80 text-[10px]">
                    {builtMeta.bands} bands · 10m grid
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 font-mono truncate">
                  {builtMeta.filename} ({builtMeta.file_size_formatted})
                </p>
                <div className="flex items-center gap-2 pt-1">
                  <a
                    href={getDownloadUrl(builtMeta.id)}
                    download={builtMeta.filename}
                    className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-mono font-semibold flex items-center gap-1.5 shadow-md shadow-emerald-950/50 transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download GeoTIFF (.tif)
                  </a>
                  <button
                    onClick={() => setIsBuilderOpen(false)}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono transition-colors"
                  >
                    Open in Studio Workspace
                  </button>
                </div>
              </div>
            )}

            <div className="flex items-center justify-between pt-3 border-t border-slate-800">
              <span className="text-[11px] text-slate-400 font-mono">
                {Object.keys(builderFiles).length >= 2
                  ? `${Object.keys(builderFiles).length} bands selected ready for alignment`
                  : "Select at least 2 single-band TIFFs to assemble"}
              </span>

              <button
                onClick={handleBuildMultispectral}
                disabled={isBuildingMultispectral || Object.keys(builderFiles).length < 2}
                className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 disabled:text-slate-600 text-white text-xs font-mono font-semibold flex items-center gap-1.5 shadow-lg shadow-cyan-950/40 transition-colors"
              >
                {isBuildingMultispectral ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Resampling & Building...
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Assemble & Load GeoTIFF
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SingleImageStudio;
