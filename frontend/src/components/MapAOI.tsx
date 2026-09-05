"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  MapPin,
  Square,
  Pentagon,
  Trash2,
  ZoomIn,
  ZoomOut,
  Crosshair,
  Search,
  CheckCircle2,
  Layers,
  Sparkles,
} from "lucide-react";
import { cn } from "../lib/utils";

interface MapAOIProps {
  onAoiDefined: (coords: [number, number][], areaKm2: number, locationName: string) => void;
}

export const MapAOI: React.FC<MapAOIProps> = ({ onAoiDefined }) => {
  const [searchQuery, setSearchQuery] = useState("Chennai Port, Tamil Nadu");
  const [center, setCenter] = useState<{ lat: number; lng: number }>({ lat: 13.0827, lng: 80.2707 });
  const [zoom, setZoom] = useState(12);
  const [drawMode, setDrawMode] = useState<"none" | "rectangle" | "polygon">("rectangle");
  const [aoiPoints, setAoiPoints] = useState<Array<{ x: number; y: number }>>([
    { x: 260, y: 160 },
    { x: 540, y: 160 },
    { x: 540, y: 380 },
    { x: 260, y: 380 },
  ]);
  const [cursorCoords, setCursorCoords] = useState({ lat: 13.0827, lng: 80.2707 });
  const [calculatedArea, setCalculatedArea] = useState(48.65);
  const [applied, setApplied] = useState(false);

  const canvasRef = useRef<HTMLDivElement>(null);

  // Preset location quick jumps
  const quickLocations = [
    { name: "Chennai Port", lat: 13.0827, lng: 80.2707, area: 48.65 },
    { name: "Mumbai Offshore", lat: 18.922, lng: 72.8347, area: 62.4 },
    { name: "Bangalore Tech Hub", lat: 12.9716, lng: 77.5946, area: 36.1 },
    { name: "Brahmaputra Basin", lat: 26.1445, lng: 91.7362, area: 85.2 },
    { name: "Delhi NCR", lat: 28.6139, lng: 77.209, area: 54.0 },
  ];

  const handleLocationSelect = (loc: typeof quickLocations[0]) => {
    setSearchQuery(loc.name);
    setCenter({ lat: loc.lat, lng: loc.lng });
    setCalculatedArea(loc.area);
    setApplied(false);
  };

  const handleClearAOI = () => {
    setAoiPoints([]);
    setCalculatedArea(0);
    setApplied(false);
  };

  const handleResetDefaultRectangle = () => {
    setAoiPoints([
      { x: 260, y: 160 },
      { x: 540, y: 160 },
      { x: 540, y: 380 },
      { x: 260, y: 380 },
    ]);
    setCalculatedArea(48.65);
    setDrawMode("rectangle");
    setApplied(false);
  };

  const handleApplyAOI = () => {
    // Generate realistic geographic coordinates from center and area
    const halfSpan = 0.05;
    const coords: [number, number][] = [
      [center.lng - halfSpan, center.lat - halfSpan],
      [center.lng + halfSpan, center.lat - halfSpan],
      [center.lng + halfSpan, center.lat + halfSpan],
      [center.lng - halfSpan, center.lat + halfSpan],
      [center.lng - halfSpan, center.lat - halfSpan],
    ];
    onAoiDefined(coords, calculatedArea, searchQuery);
    setApplied(true);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Convert relative pixel pos to simulated lat/lng
    const latOffset = (rect.height / 2 - y) * 0.0003 * (15 / zoom);
    const lngOffset = (x - rect.width / 2) * 0.0003 * (15 / zoom);
    setCursorCoords({
      lat: Number((center.lat + latOffset).toFixed(4)),
      lng: Number((center.lng + lngOffset).toFixed(4)),
    });
  };

  return (
    <div className="space-y-4">
      {/* Search & Location Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-lg bg-slate-900/60 border border-slate-800">
        <div className="flex items-center space-x-2 flex-1 min-w-[280px]">
          <Search className="w-4 h-4 text-cyan-400 shrink-0" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search city, harbor, coordinates, or geohash..."
            className="w-full bg-slate-950 border border-slate-700/80 rounded px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500"
          />
        </div>

        {/* Quick presets */}
        <div className="flex items-center gap-1.5 overflow-x-auto text-[11px] font-mono">
          <span className="text-slate-500 text-[10px] uppercase">Presets:</span>
          {quickLocations.map((loc) => (
            <button
              key={loc.name}
              onClick={() => handleLocationSelect(loc)}
              className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors whitespace-nowrap"
            >
              {loc.name}
            </button>
          ))}
        </div>
      </div>

      {/* Interactive Map Canvas Container */}
      <div className="relative rounded-xl border border-slate-800 bg-[#090e1a] overflow-hidden select-none shadow-2xl">
        {/* Top Floating Telemetry Overlay */}
        <div className="absolute top-3 left-3 z-10 flex items-center space-x-2 text-[11px] font-mono bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-md border border-slate-700/80 text-slate-300 shadow-lg">
          <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
          <span>CURSOR:</span>
          <span className="text-emerald-400 font-semibold">{cursorCoords.lat}° N, {cursorCoords.lng}° E</span>
          <span className="text-slate-600">|</span>
          <span>ZOOM:</span>
          <span className="text-cyan-400">{zoom}x</span>
        </div>

        {/* Top Right Tool Panel */}
        <div className="absolute top-3 right-3 z-10 flex items-center space-x-1.5 bg-slate-900/90 backdrop-blur-md p-1 rounded-md border border-slate-700/80 shadow-lg">
          <button
            onClick={() => setDrawMode("rectangle")}
            title="Draw Rectangle AOI"
            className={cn(
              "px-2.5 py-1 rounded text-xs font-mono flex items-center space-x-1 transition-colors",
              drawMode === "rectangle"
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                : "text-slate-400 hover:text-slate-200"
            )}
          >
            <Square className="w-3.5 h-3.5" />
            <span>Rectangle</span>
          </button>
          <button
            onClick={() => setDrawMode("polygon")}
            title="Draw Custom Polygon AOI"
            className={cn(
              "px-2.5 py-1 rounded text-xs font-mono flex items-center space-x-1 transition-colors",
              drawMode === "polygon"
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                : "text-slate-400 hover:text-slate-200"
            )}
          >
            <Pentagon className="w-3.5 h-3.5" />
            <span>Polygon</span>
          </button>
          <button
            onClick={handleClearAOI}
            title="Clear Area of Interest"
            className="p-1 rounded text-slate-400 hover:text-rose-400 hover:bg-slate-800 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Zoom Controls */}
        <div className="absolute bottom-4 right-4 z-10 flex flex-col space-y-1 bg-slate-900/90 backdrop-blur-md p-1 rounded-md border border-slate-700/80 shadow-lg">
          <button
            onClick={() => setZoom((z) => Math.min(18, z + 1))}
            className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(4, z - 1))}
            className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={handleResetDefaultRectangle}
            title="Reset Centered AOI"
            className="p-1.5 text-slate-400 hover:text-emerald-400 hover:bg-slate-800 rounded transition-colors"
          >
            <Crosshair className="w-4 h-4" />
          </button>
        </div>

        {/* Bottom Left AOI Area Metric */}
        <div className="absolute bottom-4 left-4 z-10 flex items-center space-x-3 bg-slate-900/90 backdrop-blur-md px-3 py-2 rounded-md border border-slate-700/80 text-xs font-mono text-slate-300 shadow-lg">
          <div>
            <div className="text-[10px] text-slate-500 uppercase">APPROXIMATE AOI AREA</div>
            <div className="text-emerald-400 font-bold text-sm">
              {calculatedArea > 0 ? `${calculatedArea} km²` : "No AOI defined"}
            </div>
          </div>
          <button
            onClick={handleApplyAOI}
            disabled={calculatedArea === 0}
            className={cn(
              "px-3 py-1.5 rounded text-xs font-sans font-semibold transition-all flex items-center space-x-1.5 shadow-sm",
              applied
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                : "bg-emerald-600 hover:bg-emerald-500 text-slate-950"
            )}
          >
            {applied ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>AOI Applied</span>
              </>
            ) : (
              <span>Use This AOI</span>
            )}
          </button>
        </div>

        {/* Vector Satellite Canvas */}
        <div
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          className="w-full h-[420px] relative cursor-crosshair overflow-hidden"
        >
          <svg className="w-full h-full" viewBox="0 0 800 420">
            <defs>
              <pattern id="grid-pattern" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#172338" strokeWidth="1" />
              </pattern>
              <linearGradient id="water-grad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#081b2e" />
                <stop offset="100%" stopColor="#040e1a" />
              </linearGradient>
            </defs>

            {/* Base Ocean / Land */}
            <rect width="800" height="420" fill="#0b1322" />
            <path
              d="M 0,220 Q 200,180 380,260 T 800,240 L 800,420 L 0,420 Z"
              fill="url(#water-grad)"
            />

            {/* Land terrain topography contours */}
            <path
              d="M 60,60 Q 220,40 340,160 T 680,110"
              stroke="#1e3a5f"
              strokeWidth="2"
              fill="none"
              strokeDasharray="4 4"
            />
            <path
              d="M 100,100 Q 260,80 380,200 T 720,150"
              stroke="#1e3a5f"
              strokeWidth="2"
              fill="none"
              strokeDasharray="4 4"
            />

            {/* Coordinate Grid Overlay */}
            <rect width="800" height="420" fill="url(#grid-pattern)" />

            {/* Target Location Point */}
            <g transform="translate(400, 210)">
              <circle r="4" fill="#38bdf8" />
              <circle r="16" fill="none" stroke="#38bdf8" strokeWidth="1" opacity="0.4" className="radar-live-dot" />
              <text x="10" y="4" fill="#94a3b8" fontFamily="monospace" fontSize="10">
                {searchQuery} ({center.lat.toFixed(2)}°N, {center.lng.toFixed(2)}°E)
              </text>
            </g>

            {/* Rendered AOI Polygon / Bounding Box */}
            {aoiPoints.length > 0 && (
              <g>
                <polygon
                  points={aoiPoints.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="rgba(16, 185, 129, 0.15)"
                  stroke="#10b981"
                  strokeWidth="2"
                  strokeDasharray="6 3"
                />
                {aoiPoints.map((p, idx) => (
                  <circle key={idx} cx={p.x} cy={p.y} r="4" fill="#10b981" stroke="#064e3b" strokeWidth="1.5" />
                ))}
                {/* Area Telemetry Tag inside Box */}
                <text
                  x={aoiPoints[0].x + 12}
                  y={aoiPoints[0].y + 24}
                  fill="#6ee7b7"
                  fontFamily="monospace"
                  fontSize="11"
                  fontWeight="bold"
                >
                  AOI: {calculatedArea} km² [EPSG:4326]
                </text>
              </g>
            )}
          </svg>
        </div>
      </div>
    </div>
  );
};
