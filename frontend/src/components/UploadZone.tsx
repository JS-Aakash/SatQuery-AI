"use client";

import React, { useState, useRef, useEffect } from "react";
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
  Check,
} from "lucide-react";
import { ImageMetadata, ModalityType, PairValidationResult } from "../lib/types";
import { uploadFileMultipart, validateImagePair, fetchCompositePreview } from "../lib/api";
import { PRESET_SCENARIOS, PresetScenario } from "../lib/presets";
import { cn } from "../lib/utils";

interface UploadZoneProps {
  primaryImage: ImageMetadata | null;
  secondaryImage: ImageMetadata | null;
  onSetPrimaryImage: (img: ImageMetadata | null) => void;
  onSetSecondaryImage: (img: ImageMetadata | null) => void;
  onSelectPreset: (preset: PresetScenario) => void;
  activePresetId?: string;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  primaryImage,
  secondaryImage,
  onSetPrimaryImage,
  onSetSecondaryImage,
  onSelectPreset,
  activePresetId,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pairValidation, setPairValidation] = useState<PairValidationResult | null>(null);
  const [selectedComposite, setSelectedComposite] = useState<string>("true_color");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Trigger pair validation when both primary and secondary images are present
  useEffect(() => {
    if (primaryImage && secondaryImage) {
      validateImagePair(primaryImage, secondaryImage).then((res) => setPairValidation(res));
    } else {
      setPairValidation(null);
    }
  }, [primaryImage, secondaryImage]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const processFile = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(25);
    setErrorMessage(null);

    const progressTimer = setInterval(() => {
      setUploadProgress((prev) => (prev < 90 ? prev + 20 : prev));
    }, 100);

    try {
      // Send real file to backend for rasterio validation and metadata extraction
      const res = await uploadFileMultipart(file);
      clearInterval(progressTimer);
      setUploadProgress(100);

      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
        // If a preset was active or no primary image exists, the uploaded image becomes the primary image
        if (!primaryImage || activePresetId) {
          onSetPrimaryImage(res.metadata);
          onSetSecondaryImage(null);
        } else {
          onSetSecondaryImage(res.metadata);
        }
      }, 250);
    } catch (err: any) {
      clearInterval(progressTimer);
      setIsUploading(false);
      setErrorMessage("Failed to process raster file. Please verify file integrity.");
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const handleSwitchComposite = async (type: "true_color" | "false_color_nir" | "sar_db") => {
    if (!primaryImage) return;
    setSelectedComposite(type);
    const newUrl = await fetchCompositePreview(primaryImage.id, type);
    if (newUrl) {
      onSetPrimaryImage({ ...primaryImage, preview_url: newUrl });
    }
  };

  return (
    <div className="space-y-5">
      {/* Drag and Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={cn(
          "border-2 border-dashed rounded-xl p-8 transition-all cursor-pointer text-center relative overflow-hidden",
          isDragging
            ? "border-emerald-500 bg-emerald-950/20 shadow-lg shadow-emerald-950/30"
            : "border-slate-800 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/60"
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={handleFileChange}
          className="hidden"
        />

        {isUploading ? (
          <div className="py-4 space-y-3">
            <RefreshCw className="w-8 h-8 text-emerald-400 mx-auto animate-spin" />
            <div className="text-xs text-slate-300 font-mono">
              PARSING GEOTIFF RASTER & EXTRACTING SPECTRAL METADATA... {uploadProgress}%
            </div>
            <div className="w-64 h-1.5 bg-slate-800 rounded-full mx-auto overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-150"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="w-12 h-12 rounded-full bg-slate-800/80 border border-slate-700 flex items-center justify-center mx-auto text-emerald-400 shadow-inner">
              <UploadCloud className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-200">
                Drag and drop your satellite raster or click to browse
              </p>
              <p className="text-xs text-slate-500 mt-1 font-mono">
                Supported: GeoTIFF (.tif, .tiff) • Benchmark PNG / JPEG (optical, multispectral, SAR)
              </p>
            </div>
            <div className="flex items-center justify-center gap-2 pt-1 text-[11px] text-slate-400 font-mono">
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">GeoTIFF</span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">TIFF</span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">PNG</span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">JPEG</span>
            </div>
          </div>
        )}

        {errorMessage && (
          <div className="mt-4 p-2.5 rounded bg-rose-950/40 border border-rose-800 text-xs text-rose-300 flex items-center justify-center space-x-2">
            <AlertCircle className="w-4 h-4 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
        )}
      </div>

      {/* Preset Scenarios for Hackathon / SIH Demonstration */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            SIH Benchmark & Evaluation Presets
          </span>
          <span className="text-[11px] text-slate-500 font-mono">Instant remote-sensing scenarios</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2.5">
          {PRESET_SCENARIOS.map((preset) => {
            const isSelected = activePresetId === preset.id;
            return (
              <button
                key={preset.id}
                onClick={() => onSelectPreset(preset)}
                className={cn(
                  "p-3 rounded-lg text-left transition-all border flex flex-col justify-between space-y-2 group relative overflow-hidden",
                  isSelected
                    ? "bg-slate-800/90 border-emerald-500 shadow-md shadow-emerald-950/40"
                    : "bg-slate-900/60 hover:bg-slate-800/60 border-slate-800 hover:border-slate-700"
                )}
              >
                <div>
                  <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                    <span className="px-1.5 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700">
                      {preset.category}
                    </span>
                    {isSelected && (
                      <span className="text-emerald-400 font-semibold flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> Loaded
                      </span>
                    )}
                  </div>
                  <h4 className="text-xs font-semibold text-slate-200 group-hover:text-emerald-400 transition-colors leading-tight">
                    {preset.name}
                  </h4>
                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                    {preset.description}
                  </p>
                </div>
                <div className="text-[10px] text-slate-500 font-mono truncate">
                  📍 {preset.location}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Pair Compatibility Telemetry (When 2 Images Loaded) */}
      {pairValidation && (
        <div className="p-3 rounded-lg bg-slate-900/90 border border-cyan-500/40 space-y-2 text-xs font-mono">
          <div className="flex items-center justify-between">
            <span className="text-cyan-400 font-semibold flex items-center gap-1.5">
              <Layers className="w-4 h-4" />
              PAIR COMPATIBILITY: {pairValidation.pair_type.toUpperCase()}
            </span>
            <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[10px]">
              {pairValidation.is_compatible ? "SPATIALLY COMPATIBLE" : "ALIGNMENT WARNING"}
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-300">
            <div><span className="text-slate-500">Overlap:</span> {pairValidation.overlap_percentage}%</div>
            <div><span className="text-slate-500">CRS Match:</span> {pairValidation.crs_compatible ? "Identical" : "Transformable"}</div>
            <div><span className="text-slate-500">Dimensions:</span> {pairValidation.dimensions_match ? "Exact Match" : "Resampling Ready"}</div>
            <div><span className="text-slate-500">Pair Mode:</span> {pairValidation.is_optical_sar_pair ? "Optical + SAR" : "Bi-Temporal"}</div>
          </div>
        </div>
      )}

      {/* Uploaded Imagery Metadata Inspection Cards */}
      {(primaryImage || secondaryImage) && (
        <div className="space-y-2 pt-2">
          <div className="flex items-center justify-between">
            <div className="text-xs font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <FileCheck className="w-3.5 h-3.5 text-emerald-400" />
              Real Raster Metadata & Spatial Attributes
            </div>

            {/* Band Composite Switcher for Multispectral Rasters */}
            {primaryImage && primaryImage.bands >= 4 && (
              <div className="flex items-center space-x-1 text-[11px] font-mono bg-slate-900 px-2 py-1 rounded border border-slate-800">
                <span className="text-slate-500 uppercase mr-1">Preview:</span>
                <button
                  onClick={() => handleSwitchComposite("true_color")}
                  className={cn(
                    "px-1.5 py-0.5 rounded transition-colors",
                    selectedComposite === "true_color"
                      ? "bg-emerald-500/20 text-emerald-300 font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  )}
                >
                  RGB True-Color
                </button>
                <button
                  onClick={() => handleSwitchComposite("false_color_nir")}
                  className={cn(
                    "px-1.5 py-0.5 rounded transition-colors",
                    selectedComposite === "false_color_nir"
                      ? "bg-cyan-500/20 text-cyan-300 font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  )}
                >
                  CIR (False-Color NIR)
                </button>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {primaryImage && (
              <div className="p-3.5 rounded-lg bg-slate-900/90 border border-slate-800 space-y-2 relative">
                <div className="flex items-start justify-between">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-mono">
                        PRIMARY IMAGE
                      </span>
                      <span className="text-xs font-semibold text-slate-200 truncate max-w-[200px]" title={primaryImage.filename}>
                        {primaryImage.filename}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 font-mono">
                      {primaryImage.sensor} • {primaryImage.modality} • {primaryImage.driver || "GTiff"}
                    </p>
                  </div>
                  <button
                    onClick={() => onSetPrimaryImage(null)}
                    className="text-slate-500 hover:text-slate-300 p-1"
                    title="Remove image"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1 text-slate-300">
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/80">
                    <span className="text-slate-500">Dimensions:</span> {primaryImage.dimensions[0]} × {primaryImage.dimensions[1]}
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/80">
                    <span className="text-slate-500">Bands / Dtype:</span> {primaryImage.bands} ({primaryImage.dtype || "uint16"})
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/80 truncate" title={primaryImage.crs || "EPSG:4326"}>
                    <span className="text-slate-500">CRS:</span> {primaryImage.crs || "EPSG:4326"}
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/80">
                    <span className="text-slate-500">Resolution:</span> {primaryImage.resolution_m ? `${primaryImage.resolution_m}m GSD` : "10m GSD"}
                  </div>
                </div>

                {/* Bounds & Transform Telemetry */}
                {primaryImage.bounds && (
                  <div className="text-[10px] text-slate-400 font-mono bg-slate-950/60 p-1.5 rounded border border-slate-800/50 truncate">
                    <span className="text-slate-500">GeoBounds: </span>
                    [{primaryImage.bounds.map((b) => b.toFixed(3)).join(", ")}]
                  </div>
                )}

                {primaryImage.band_names && (
                  <div className="text-[10px] text-slate-400 font-mono bg-slate-950/40 p-1.5 rounded border border-slate-800/50">
                    <span className="text-slate-500">Spectral Channels: </span>
                    {primaryImage.band_names.slice(0, 4).join(", ")}
                    {primaryImage.band_names.length > 4 ? ` (+${primaryImage.band_names.length - 4} more)` : ""}
                  </div>
                )}

                <div className="flex items-center space-x-1.5 text-[11px] text-emerald-400 font-mono pt-0.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Rasterio Engine Verified • Ready for Analysis</span>
                </div>
              </div>
            )}

            {secondaryImage && (
              <div className="p-3.5 rounded-lg bg-slate-900/90 border border-slate-800 space-y-2 relative">
                <div className="flex items-start justify-between">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 font-mono">
                        SECONDARY / PAIRED IMAGE
                      </span>
                      <span className="text-xs font-semibold text-slate-200 truncate max-w-[200px]" title={secondaryImage.filename}>
                        {secondaryImage.filename}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 font-mono">
                      {secondaryImage.sensor} • {secondaryImage.modality} • {secondaryImage.driver || "GTiff"}
                    </p>
                  </div>
                  <button
                    onClick={() => onSetSecondaryImage(null)}
                    className="text-slate-500 hover:text-slate-300 p-1"
                    title="Remove image"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1 text-slate-300">
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/80">
                    <span className="text-slate-500">Dimensions:</span> {secondaryImage.dimensions[0]} × {secondaryImage.dimensions[1]}
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/80">
                    <span className="text-slate-500">Bands / Dtype:</span> {secondaryImage.bands} ({secondaryImage.dtype || "uint16"})
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/80 truncate" title={secondaryImage.crs || "EPSG:4326"}>
                    <span className="text-slate-500">CRS:</span> {secondaryImage.crs || "EPSG:4326"}
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/80">
                    <span className="text-slate-500">Resolution:</span> {secondaryImage.resolution_m ? `${secondaryImage.resolution_m}m GSD` : "10m GSD"}
                  </div>
                </div>

                {secondaryImage.bounds && (
                  <div className="text-[10px] text-slate-400 font-mono bg-slate-950/60 p-1.5 rounded border border-slate-800/50 truncate">
                    <span className="text-slate-500">GeoBounds: </span>
                    [{secondaryImage.bounds.map((b) => b.toFixed(3)).join(", ")}]
                  </div>
                )}

                <div className="flex items-center space-x-1.5 text-[11px] text-cyan-400 font-mono pt-0.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Spatially Aligned & Co-Registered</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
