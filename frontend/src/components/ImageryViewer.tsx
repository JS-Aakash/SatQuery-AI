"use client";

import React, { useState, useRef } from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  Sliders,
  Layers,
  Split,
  Columns,
  Eye,
  EyeOff,
  RotateCcw,
  Sparkles,
  Square,
  Crosshair,
} from "lucide-react";
import { GroundingBox, ChangeMapData } from "../lib/types";
import { cn } from "../lib/utils";

interface ImageryViewerProps {
  imageAUrl: string;
  imageBUrl?: string;
  imageAName: string;
  imageBName?: string;
  groundingBoxes?: GroundingBox[];
  changeMap?: ChangeMapData | null;
  hasTemporalComparison?: boolean;
  selectedBoxId?: string | null;
  onSelectBox?: (id: string | null) => void;
}

export const ImageryViewer: React.FC<ImageryViewerProps> = ({
  imageAUrl,
  imageBUrl,
  imageAName,
  imageBName,
  groundingBoxes = [],
  changeMap,
  hasTemporalComparison = false,
  selectedBoxId,
  onSelectBox,
}) => {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Comparison controls
  const [comparisonMode, setComparisonMode] = useState<"split" | "side-by-side" | "single">(
    hasTemporalComparison && imageBUrl ? "split" : "single"
  );
  const [splitPosition, setSplitPosition] = useState(50); // percentage 0 - 100

  // Layer visibility & opacity
  const [showGrounding, setShowGrounding] = useState(true);
  const [showChangeMap, setShowChangeMap] = useState(true);
  const [changeMapOpacity, setChangeMapOpacity] = useState(0.7);
  const [activeBoxId, setActiveBoxId] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const sliderBoxRef = useRef<HTMLDivElement>(null);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);

  const handleZoomIn = () => setZoom((z) => Math.min(4, Number((z + 0.25).toFixed(2))));
  const handleZoomOut = () => setZoom((z) => Math.max(0.5, Number((z - 0.25).toFixed(2))));
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setSplitPosition(50);
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!isFullscreen) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen();
      }
      setIsFullscreen(true);
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
      setIsFullscreen(false);
    }
  };

  const updateSplitFromPointer = (clientX: number) => {
    if (!sliderBoxRef.current) return;
    const rect = sliderBoxRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const rawPct = ((clientX - rect.left) / rect.width) * 100;
    const clamped = Math.max(0, Math.min(100, Math.round(rawPct * 10) / 10));
    setSplitPosition(clamped);
  };

  const handleSliderPointerDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    setIsDraggingSlider(true);
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {}
    updateSplitFromPointer(e.clientX);
  };

  const handleSliderPointerMove = (e: React.PointerEvent) => {
    if (!isDraggingSlider) return;
    e.stopPropagation();
    updateSplitFromPointer(e.clientX);
  };

  const handleSliderPointerUp = (e: React.PointerEvent) => {
    if (isDraggingSlider) {
      e.stopPropagation();
      setIsDraggingSlider(false);
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {}
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (isDraggingSlider) return;
    setIsPanning(true);
    setStartPan({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning || isDraggingSlider) return;
    setPan({ x: e.clientX - startPan.x, y: e.clientY - startPan.y });
  };

  const handleMouseUp = () => setIsPanning(false);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative rounded-xl border border-slate-800 bg-[#060a12] overflow-hidden flex flex-col select-none shadow-2xl transition-all",
        isFullscreen ? "fixed inset-0 z-50 rounded-none h-screen" : "h-[620px] w-full"
      )}
    >
      {/* Top Floating Viewport Toolbar */}
      <div className="absolute top-3 left-3 right-3 z-20 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* Left Telemetry HUD */}
        <div className="flex items-center space-x-2 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-md border border-slate-700/80 text-[11px] font-mono text-slate-300 pointer-events-auto shadow-lg">
          <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
          <span className="truncate max-w-[200px] text-slate-200">{imageAName}</span>
          {imageBUrl && (
            <>
              <span className="text-slate-600">vs</span>
              <span className="truncate max-w-[180px] text-cyan-300">{imageBName || "Observation T2"}</span>
            </>
          )}
          <span className="text-slate-600">|</span>
          <span>ZOOM:</span>
          <span className="text-emerald-400">{Math.round(zoom * 100)}%</span>
        </div>

        {/* Right Action Tools */}
        <div className="flex items-center space-x-2 pointer-events-auto">
          {/* Comparison Mode Toggles if secondary image exists */}
          {imageBUrl && (
            <div className="flex items-center space-x-1 bg-slate-900/90 backdrop-blur-md p-1 rounded-md border border-slate-700/80 text-xs shadow-lg font-mono">
              <button
                onClick={() => setComparisonMode("split")}
                className={cn(
                  "px-2 py-1 rounded flex items-center space-x-1 transition-colors",
                  comparisonMode === "split"
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    : "text-slate-400 hover:text-slate-200"
                )}
                title="Split swipe slider"
              >
                <Split className="w-3.5 h-3.5" />
                <span className="text-[11px]">Swipe</span>
              </button>
              <button
                onClick={() => setComparisonMode("side-by-side")}
                className={cn(
                  "px-2 py-1 rounded flex items-center space-x-1 transition-colors",
                  comparisonMode === "side-by-side"
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    : "text-slate-400 hover:text-slate-200"
                )}
                title="Side by side comparison"
              >
                <Columns className="w-3.5 h-3.5" />
                <span className="text-[11px]">Dual</span>
              </button>
            </div>
          )}

          {/* Layer Toggles */}
          <div className="flex items-center space-x-1 bg-slate-900/90 backdrop-blur-md p-1 rounded-md border border-slate-700/80 text-xs shadow-lg font-mono">
            {groundingBoxes.length > 0 && (
              <button
                onClick={() => setShowGrounding(!showGrounding)}
                className={cn(
                  "px-2 py-1 rounded flex items-center space-x-1 transition-colors",
                  showGrounding
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                    : "text-slate-500 hover:text-slate-300"
                )}
                title="Toggle text grounding bounding boxes"
              >
                <Square className="w-3 h-3" />
                <span className="text-[11px]">Boxes ({groundingBoxes.length})</span>
              </button>
            )}

            {changeMap?.has_change && (
              <button
                onClick={() => setShowChangeMap(!showChangeMap)}
                className={cn(
                  "px-2 py-1 rounded flex items-center space-x-1 transition-colors",
                  showChangeMap
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                    : "text-slate-500 hover:text-slate-300"
                )}
                title="Toggle spatial change detection heatmap"
              >
                <Layers className="w-3 h-3" />
                <span className="text-[11px]">Change Map</span>
              </button>
            )}
          </div>

          {/* Zoom & Fullscreen Tools */}
          <div className="flex items-center space-x-1 bg-slate-900/90 backdrop-blur-md p-1 rounded-md border border-slate-700/80 shadow-lg">
            <button
              onClick={handleZoomIn}
              className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={handleReset}
              className="p-1.5 text-slate-400 hover:text-emerald-400 hover:bg-slate-800 rounded transition-colors"
              title="Reset View"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              onClick={toggleFullscreen}
              className="p-1.5 text-slate-400 hover:text-cyan-400 hover:bg-slate-800 rounded transition-colors"
              title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Main Imagery Canvas Area */}
      <div
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={cn(
          "flex-1 relative overflow-hidden flex items-center justify-center cursor-grab active:cursor-grabbing bg-slate-950",
          isPanning ? "cursor-grabbing" : "cursor-grab"
        )}
      >
        <div
          className="relative transition-transform duration-75 origin-center w-full h-full flex items-center justify-center"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          }}
        >
          {/* Case 1: Split-Screen Swipe Slider */}
          {comparisonMode === "split" && imageBUrl ? (
            <div
              ref={sliderBoxRef}
              onPointerDown={handleSliderPointerDown}
              onPointerMove={handleSliderPointerMove}
              onPointerUp={handleSliderPointerUp}
              onPointerCancel={handleSliderPointerUp}
              className="relative w-full h-full max-w-[860px] max-h-[580px] aspect-[4/3] rounded-md overflow-hidden border border-slate-800 shadow-2xl select-none touch-none cursor-ew-resize"
            >
              {/* Layer B (Right side of swipe) */}
              <img
                src={imageBUrl}
                alt={imageBName || "Image B"}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none select-none"
                draggable={false}
              />
              <div className="absolute bottom-3 right-3 px-2 py-1 rounded bg-slate-900/90 text-cyan-300 font-mono text-[10px] border border-slate-700 pointer-events-none select-none">
                {imageBName || "OBSERVATION T2 / SAR"}
              </div>

              {/* Layer A (Left side of swipe, clipped) */}
              <div
                className="absolute inset-0 overflow-hidden pointer-events-none select-none"
                style={{ clipPath: `polygon(0 0, ${splitPosition}% 0, ${splitPosition}% 100%, 0 100%)` }}
              >
                <img
                  src={imageAUrl}
                  alt={imageAName}
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none select-none"
                  draggable={false}
                />
                <div className="absolute bottom-3 left-3 px-2 py-1 rounded bg-slate-900/90 text-emerald-300 font-mono text-[10px] border border-slate-700 pointer-events-none select-none">
                  {imageAName || "OBSERVATION T1 / OPTICAL"}
                </div>
              </div>

              {/* Change Map Heatmap Overlay in Swipe Mode */}
              {showChangeMap && changeMap?.has_change && (
                <div
                  className="absolute inset-0 pointer-events-none transition-opacity duration-150 z-10"
                  style={{ opacity: changeMapOpacity }}
                >
                  {changeMap.change_mask_url ? (
                    <img
                      src={changeMap.change_mask_url}
                      alt="Change Mask"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <svg className="w-full h-full" viewBox="0 0 800 600">
                      <rect x="60" y="60" width="300" height="250" fill="#f59e0b" opacity="0.6" stroke="#fbbf24" strokeWidth="2" />
                      <path d="M 250,450 Q 250,340 420,340 Q 590,340 590,450 Q 590,560 420,560 Q 250,560 250,450 Z" fill="#ef4444" opacity="0.5" />
                      <path d="M 20,310 Q 300,330 780,360" stroke="#06b6d4" strokeWidth="10" fill="none" opacity="0.8" />
                    </svg>
                  )}
                </div>
              )}

              {/* Split Slider Divider Line & Handle */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.9)] z-20 flex items-center justify-center pointer-events-none"
                style={{ left: `${splitPosition}%` }}
              >
                <div className="w-8 h-8 rounded-full bg-slate-900 border-2 border-emerald-400 flex items-center justify-center shadow-xl -translate-x-1/2">
                  <Split className="w-4 h-4 text-emerald-400" />
                </div>
              </div>

              {/* Text-Guided Grounding Bounding Boxes Overlay on Swipe Canvas */}
              {showGrounding && groundingBoxes.length > 0 && (
                <svg className="absolute inset-0 w-full h-full pointer-events-none z-10" viewBox="0 0 100 100" preserveAspectRatio="none">
                  {groundingBoxes.map((gb) => {
                    const [ymin, xmin, ymax, xmax] = gb.box;
                    const isHovered = activeBoxId === gb.id;
                    const isSelected = selectedBoxId === gb.id;
                    const isFocal = isHovered || isSelected;
                    const strokeColor = gb.color || "#10b981";

                    return (
                      <g
                        key={`swipe-${gb.id}`}
                        onMouseEnter={() => {
                          setActiveBoxId(gb.id);
                          onSelectBox?.(gb.id);
                        }}
                        onMouseLeave={() => {
                          setActiveBoxId(null);
                        }}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectBox?.(isSelected ? null : gb.id);
                        }}
                        className="cursor-pointer group pointer-events-auto"
                      >
                        <rect
                          x={xmin}
                          y={ymin}
                          width={xmax - xmin}
                          height={ymax - ymin}
                          fill={isFocal ? "rgba(16, 185, 129, 0.28)" : "rgba(16, 185, 129, 0.08)"}
                          stroke={isFocal ? "#34d399" : strokeColor}
                          strokeWidth={isFocal ? "0.9" : "0.5"}
                          strokeDasharray={isFocal ? undefined : "1.5 1"}
                          className="transition-all duration-150"
                        />
                        <g transform={`translate(${xmin}, ${Math.max(3, ymin - 1)})`}>
                          <rect
                            x="0"
                            y="-2.5"
                            width={gb.label.length * 1.4 + 9}
                            height="3.5"
                            fill={isFocal ? "#064e3b" : "#0b1322"}
                            stroke={isFocal ? "#34d399" : strokeColor}
                            strokeWidth={isFocal ? "0.5" : "0.3"}
                            rx="0.5"
                          />
                          <text
                            x="1"
                            y="0"
                            fill="#f8fafc"
                            fontSize="2.2"
                            fontFamily="monospace"
                            fontWeight="bold"
                          >
                            {gb.label} ({Math.round(gb.confidence * 100)}%)
                          </text>
                        </g>
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          ) : comparisonMode === "side-by-side" && imageBUrl ? (
            /* Case 2: Side by Side Dual View */
            <div className="grid grid-cols-2 gap-3 w-full h-full max-w-[940px] max-h-[580px] p-4">
              <div className="relative rounded border border-slate-800 overflow-hidden flex items-center justify-center bg-slate-950">
                <img src={imageAUrl} alt={imageAName} className="w-full h-full object-contain" draggable={false} />
                <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-slate-900/90 text-emerald-400 font-mono text-[10px] border border-slate-700">
                  {imageAName}
                </div>
              </div>
              <div className="relative rounded border border-slate-800 overflow-hidden flex items-center justify-center bg-slate-950">
                <img src={imageBUrl} alt={imageBName || "Image B"} className="w-full h-full object-contain" draggable={false} />
                <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-slate-900/90 text-cyan-400 font-mono text-[10px] border border-slate-700">
                  {imageBName || "Observation B"}
                </div>
              </div>
            </div>
          ) : (
            /* Case 3: Single Primary Image with SVG Overlays */
            <div className="relative w-full h-full max-w-[860px] max-h-[580px] rounded-md overflow-hidden border border-slate-800 shadow-2xl flex items-center justify-center bg-[#070b14]">
              <div className="relative w-full h-full flex items-center justify-center">
                <img
                  src={imageAUrl}
                  alt={imageAName}
                  className="w-full h-full object-contain pointer-events-none select-none"
                  draggable={false}
                />

                {/* Change Map Heatmap Overlay */}
                {showChangeMap && changeMap?.has_change && (
                  <div
                    className="absolute inset-0 pointer-events-none transition-opacity duration-150 z-10"
                    style={{ opacity: changeMapOpacity }}
                  >
                    {changeMap.change_mask_url ? (
                      <img
                        src={changeMap.change_mask_url}
                        alt="Change Mask"
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <svg className="w-full h-full" viewBox="0 0 800 600">
                        <rect x="60" y="60" width="300" height="250" fill="#f59e0b" opacity="0.6" stroke="#fbbf24" strokeWidth="2" />
                        <path d="M 250,450 Q 250,340 420,340 Q 590,340 590,450 Q 590,560 420,560 Q 250,560 250,450 Z" fill="#ef4444" opacity="0.5" />
                        <path d="M 20,310 Q 300,330 780,360" stroke="#06b6d4" strokeWidth="10" fill="none" opacity="0.8" />
                      </svg>
                    )}
                  </div>
                )}

                {/* Text-Guided Grounding Bounding Boxes Overlay */}
                {showGrounding && groundingBoxes.length > 0 && (
                  <svg className="absolute inset-0 w-full h-full pointer-events-auto" viewBox="0 0 100 100" preserveAspectRatio="none">
                    {groundingBoxes.map((gb) => {
                      const [ymin, xmin, ymax, xmax] = gb.box;
                      const isHovered = activeBoxId === gb.id;
                      const isSelected = selectedBoxId === gb.id;
                      const isFocal = isHovered || isSelected;
                      const strokeColor = gb.color || "#10b981";

                      return (
                        <g
                          key={gb.id}
                          onMouseEnter={() => {
                            setActiveBoxId(gb.id);
                            onSelectBox?.(gb.id);
                          }}
                          onMouseLeave={() => {
                            setActiveBoxId(null);
                          }}
                          onClick={() => onSelectBox?.(isSelected ? null : gb.id)}
                          className="cursor-pointer group"
                        >
                          <rect
                            x={xmin}
                            y={ymin}
                            width={xmax - xmin}
                            height={ymax - ymin}
                            fill={isFocal ? "rgba(16, 185, 129, 0.28)" : "rgba(16, 185, 129, 0.08)"}
                            stroke={isFocal ? "#34d399" : strokeColor}
                            strokeWidth={isFocal ? "0.9" : "0.5"}
                            strokeDasharray={isFocal ? undefined : "1.5 1"}
                            className="transition-all duration-150"
                          />
                          {/* Text Tag */}
                          <g transform={`translate(${xmin}, ${Math.max(3, ymin - 1)})`}>
                            <rect
                              x="0"
                              y="-2.5"
                              width={gb.label.length * 1.4 + 9}
                              height="3.5"
                              fill={isFocal ? "#064e3b" : "#0b1322"}
                              stroke={isFocal ? "#34d399" : strokeColor}
                              strokeWidth={isFocal ? "0.5" : "0.3"}
                              rx="0.5"
                            />
                            <text
                              x="1"
                              y="0"
                              fill="#f8fafc"
                              fontSize="2.2"
                              fontFamily="monospace"
                              fontWeight="bold"
                            >
                              {gb.label} ({Math.round(gb.confidence * 100)}%)
                            </text>
                          </g>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Floating Legend & Controls */}
      <div className="absolute bottom-3 left-3 right-3 z-20 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* Change Map Legend */}
        {showChangeMap && changeMap?.legend && (
          <div className="flex items-center space-x-3 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-md border border-slate-700/80 text-[10px] font-mono text-slate-300 pointer-events-auto shadow-lg">
            <span className="text-slate-500 uppercase">CHANGE LEGEND:</span>
            {Object.entries(changeMap.legend).map(([label, color]) => {
              if (color === "transparent") return null;
              return (
                <div key={label} className="flex items-center space-x-1">
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: color }} />
                  <span>{label}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Change Map Opacity Slider */}
        {showChangeMap && changeMap?.has_change && (
          <div className="flex items-center space-x-2 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-md border border-slate-700/80 text-[11px] font-mono text-slate-300 pointer-events-auto shadow-lg ml-auto">
            <Sliders className="w-3.5 h-3.5 text-amber-400" />
            <span>OPACITY:</span>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={changeMapOpacity}
              onChange={(e) => setChangeMapOpacity(Number(e.target.value))}
              className="w-20 h-1 bg-slate-700 rounded appearance-none cursor-pointer accent-amber-400"
            />
            <span>{Math.round(changeMapOpacity * 100)}%</span>
          </div>
        )}
      </div>
    </div>
  );
};
