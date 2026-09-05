"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  MapPin,
  Layers,
  Square,
  Maximize2,
  Minimize2,
  Trash2,
  Eye,
  Sliders,
  Sparkles,
  Info,
  Compass
} from "lucide-react";
import { CopernicusAOI } from "../lib/types";

interface MapVectorPolygon {
  id: string;
  label: string;
  coordinates: number[][];
  color?: string;
  area_ha?: number;
}

interface SatelliteMapProps {
  center: [number, number]; // [lat, lon]
  zoom?: number;
  aoi?: CopernicusAOI;
  onAOIChange?: (newAoi: CopernicusAOI, areaKm2: number) => void;
  overlayImageUrl?: string;
  overlayLegend?: Record<string, string>;
  overlayTitle?: string;
  groundingBoxes?: Array<{ id: string; label: string; box: [number, number, number, number]; color?: string }>;
  areaKm2?: number;
  polygons?: MapVectorPolygon[];
  selectedPolygonId?: string;
  onSelectPolygon?: (poly: MapVectorPolygon) => void;
}

export default function SatelliteMap({
  center,
  zoom = 13,
  aoi,
  onAOIChange,
  overlayImageUrl,
  overlayLegend,
  overlayTitle,
  groundingBoxes,
  areaKm2 = 0,
  polygons,
  selectedPolygonId,
  onSelectPolygon,
}: SatelliteMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const aoiLayerRef = useRef<any>(null);
  const overlayLayerRef = useRef<any>(null);
  const polygonsLayerGroupRef = useRef<any>(null);
  const boxesLayerGroupRef = useRef<any>(null);
  const markerRef = useRef<any>(null);


  const [drawMode, setDrawMode] = useState<"none" | "rectangle" | "polygon">("none");
  const [baseMap, setBaseMap] = useState<"osm" | "satellite">("satellite");
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.85);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [mapLoaded, setMapLoaded] = useState<boolean>(false);
  const [polyPoints, setPolyPoints] = useState<[number, number][]>([]);

  // Dynamically load Leaflet on client side
  useEffect(() => {
    if (typeof window === "undefined" || !mapContainerRef.current) return;

    let isMounted = true;

    async function initLeafletMap() {
      const L = (await import("leaflet")).default;
      // Inject leaflet CSS if not present
      if (!document.getElementById("leaflet-css")) {
        const link = document.createElement("link");
        link.id = "leaflet-css";
        link.rel = "stylesheet";
        link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
        document.head.appendChild(link);
      }

      if (!mapContainerRef.current || mapInstanceRef.current) return;

      const map = L.map(mapContainerRef.current, {
        center: [center[0], center[1]],
        zoom: zoom,
        zoomControl: false,
      });

      // Add zoom control top-right
      L.control.zoom({ position: "topright" }).addTo(map);

      // Base tile layers
      const osmLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        maxZoom: 19,
      });

      const esriSatelliteLayer = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
          attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
          maxZoom: 18,
        }
      );

      if (baseMap === "satellite") {
        esriSatelliteLayer.addTo(map);
      } else {
        osmLayer.addTo(map);
      }

      mapInstanceRef.current = map;
      (map as any)._osmLayer = osmLayer;
      (map as any)._satelliteLayer = esriSatelliteLayer;

      // Center marker
      const marker = L.circleMarker([center[0], center[1]], {
        radius: 6,
        color: "#38bdf8",
        fillColor: "#0284c7",
        fillOpacity: 0.9,
      }).addTo(map);
      markerRef.current = marker;

      // Draw initial AOI Polygon
      if (aoi && aoi.coordinates && aoi.coordinates[0]) {
        const latLngs = aoi.coordinates[0].map((coord) => [coord[1], coord[0]]);
        const aoiPolygon = L.polygon(latLngs as any, {
          color: "#10b981",
          weight: 2,
          fillColor: "#10b981",
          fillOpacity: 0.15,
          dashArray: "4, 4",
        }).addTo(map);
        aoiLayerRef.current = aoiPolygon;
      }

      setMapLoaded(true);

      // Map click handler for drawing
      map.on("click", (e: any) => {
        const lat = e.latlng.lat;
        const lng = e.latlng.lng;

        if ((window as any)._currentDrawMode === "polygon") {
          const newPts = [...((window as any)._polyPoints || []), [lat, lng]];
          (window as any)._polyPoints = newPts;
          setPolyPoints(newPts as any);

          if (aoiLayerRef.current) {
            map.removeLayer(aoiLayerRef.current);
          }

          const poly = L.polygon(newPts as any, {
            color: "#f59e0b",
            weight: 2,
            fillColor: "#f59e0b",
            fillOpacity: 0.2,
          }).addTo(map);
          aoiLayerRef.current = poly;

          // If polygon has 4+ points, commit AOI
          if (newPts.length >= 4) {
            const coords = [...newPts.map((p) => [p[1], p[0]]), [newPts[0][1], newPts[0][0]]];
            const newAoi: CopernicusAOI = {
              type: "Polygon",
              coordinates: [coords],
            };
            const computedArea = calculatePolygonAreaKm2(coords);
            if (onAOIChange) {
              onAOIChange(newAoi, computedArea);
            }
            setDrawMode("none");
            (window as any)._currentDrawMode = "none";
          }
        }
      });
    }

    initLeafletMap();

    return () => {
      isMounted = false;
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update center when location changes
  useEffect(() => {
    if (mapInstanceRef.current && mapLoaded) {
      mapInstanceRef.current.setView([center[0], center[1]], zoom, { animate: true });
      if (markerRef.current) {
        markerRef.current.setLatLng([center[0], center[1]]);
      }
    }
  }, [center, zoom, mapLoaded]);

  // Update base map layer switch
  useEffect(() => {
    if (!mapInstanceRef.current || !mapLoaded) return;
    const map = mapInstanceRef.current;
    const osm = map._osmLayer;
    const sat = map._satelliteLayer;

    if (baseMap === "satellite") {
      if (map.hasLayer(osm)) map.removeLayer(osm);
      if (!map.hasLayer(sat)) sat.addTo(map);
    } else {
      if (map.hasLayer(sat)) map.removeLayer(sat);
      if (!map.hasLayer(osm)) osm.addTo(map);
    }
  }, [baseMap, mapLoaded]);

  // Update AOI Polygon layer on map
  useEffect(() => {
    if (!mapInstanceRef.current || !mapLoaded) return;
    const map = mapInstanceRef.current;

    (async () => {
      const L = (await import("leaflet")).default;
      if (aoiLayerRef.current) {
        map.removeLayer(aoiLayerRef.current);
        aoiLayerRef.current = null;
      }

      if (aoi && aoi.coordinates && aoi.coordinates[0] && aoi.coordinates[0].length >= 3) {
        const latLngs = aoi.coordinates[0].map((coord) => [coord[1], coord[0]]);
        const aoiPolygon = L.polygon(latLngs as any, {
          color: overlayImageUrl ? "#38bdf8" : "#10b981",
          weight: 2,
          fillColor: overlayImageUrl ? "transparent" : "#10b981",
          fillOpacity: overlayImageUrl ? 0 : 0.18,
          dashArray: "4, 4",
        }).addTo(map);
        aoiLayerRef.current = aoiPolygon;
      }
    })();
  }, [aoi, overlayImageUrl, mapLoaded]);

  // Update Satellite Raster Image Overlay
  useEffect(() => {
    if (!mapInstanceRef.current || !mapLoaded) return;
    const map = mapInstanceRef.current;

    (async () => {
      const L = (await import("leaflet")).default;
      if (overlayLayerRef.current) {
        map.removeLayer(overlayLayerRef.current);
        overlayLayerRef.current = null;
      }

      if (overlayImageUrl && aoi && aoi.coordinates && aoi.coordinates[0]) {
        const coords = aoi.coordinates[0];
        const lats = coords.map((c) => c[1]);
        const lons = coords.map((c) => c[0]);
        const bounds: [[number, number], [number, number]] = [
          [Math.min(...lats), Math.min(...lons)],
          [Math.max(...lats), Math.max(...lons)],
        ];

        const imageOverlay = L.imageOverlay(overlayImageUrl, bounds, {
          opacity: overlayOpacity,
          interactive: false,
        }).addTo(map);

        overlayLayerRef.current = imageOverlay;
        map.fitBounds(bounds, { padding: [30, 30] });
      }
    })();
  }, [overlayImageUrl, overlayOpacity, aoi, mapLoaded]);

  // Update Grounding Boxes Overlay
  useEffect(() => {
    if (!mapInstanceRef.current || !mapLoaded) return;
    const map = mapInstanceRef.current;

    (async () => {
      const L = (await import("leaflet")).default;
      if (boxesLayerGroupRef.current) {
        map.removeLayer(boxesLayerGroupRef.current);
        boxesLayerGroupRef.current = null;
      }

      if (groundingBoxes && groundingBoxes.length > 0 && aoi && aoi.coordinates && aoi.coordinates[0]) {
        const coords = aoi.coordinates[0];
        const lats = coords.map((c) => c[1]);
        const lons = coords.map((c) => c[0]);
        const minLat = Math.min(...lats);
        const maxLat = Math.max(...lats);
        const minLon = Math.min(...lons);
        const maxLon = Math.max(...lons);
        const dLat = maxLat - minLat;
        const dLon = maxLon - minLon;

        const group = L.layerGroup();

        groundingBoxes.forEach((gb) => {
          if (!gb.box || gb.box.length !== 4) return;
          const [ymin, xmin, ymax, xmax] = gb.box;
          const boxMinLat = maxLat - (ymax / 100.0) * dLat;
          const boxMaxLat = maxLat - (ymin / 100.0) * dLat;
          const boxMinLon = minLon + (xmin / 100.0) * dLon;
          const boxMaxLon = minLon + (xmax / 100.0) * dLon;

          const rect = L.rectangle(
            [
              [boxMinLat, boxMinLon],
              [boxMaxLat, boxMaxLon],
            ],
            {
              color: gb.color || "#f59e0b",
              weight: 2.5,
              fillColor: gb.color || "#f59e0b",
              fillOpacity: 0.18,
              dashArray: "3, 3",
            }
          );

          if (gb.label) {
            rect.bindTooltip(gb.label, {
              permanent: true,
              direction: "top",
              className: "leaflet-grounding-tooltip",
            });
          }

          group.addLayer(rect);
        });

        group.addTo(map);
        boxesLayerGroupRef.current = group;
      }
    })();
  }, [groundingBoxes, aoi, mapLoaded]);

  // Update Vector Polygons Layer (GeoJSON / Spatial grounding polygons)
  useEffect(() => {
    if (!mapInstanceRef.current || !mapLoaded) return;
    const map = mapInstanceRef.current;

    (async () => {
      const L = (await import("leaflet")).default;
      if (polygonsLayerGroupRef.current) {
        map.removeLayer(polygonsLayerGroupRef.current);
        polygonsLayerGroupRef.current = null;
      }

      if (polygons && polygons.length > 0) {
        const group = L.layerGroup();
        let selectedLayer: any = null;

        polygons.forEach((poly) => {
          if (!poly.coordinates || poly.coordinates.length < 3) return;
          const latLngs = poly.coordinates.map((c) => [c[1], c[0]]);
          const isSelected = poly.id === selectedPolygonId;

          const leafletPoly = L.polygon(latLngs as any, {
            color: isSelected ? "#38bdf8" : (poly.color || "#10b981"),
            weight: isSelected ? 3.5 : 2,
            fillColor: isSelected ? "#38bdf8" : (poly.color || "#10b981"),
            fillOpacity: isSelected ? 0.45 : 0.25,
          });

          if (poly.label) {
            leafletPoly.bindTooltip(
              `<strong>${poly.label}</strong>${poly.area_ha ? ` (${poly.area_ha.toFixed(2)} ha)` : ""}`,
              {
                permanent: isSelected,
                direction: "top",
                className: "leaflet-grounding-tooltip",
              }
            );
          }

          leafletPoly.on("click", () => {
            if (onSelectPolygon) {
              onSelectPolygon(poly);
            }
          });

          if (isSelected) {
            selectedLayer = leafletPoly;
          }

          group.addLayer(leafletPoly);
        });

        group.addTo(map);
        polygonsLayerGroupRef.current = group;

        if (selectedLayer) {
          try {
            map.fitBounds(selectedLayer.getBounds(), { maxZoom: 16, padding: [40, 40] });
          } catch (e) {
            // ignore bounds fit error
          }
        }
      }
    })();
  }, [polygons, selectedPolygonId, onSelectPolygon, mapLoaded]);


  // Sync drawMode with window global
  useEffect(() => {
    (window as any)._currentDrawMode = drawMode;
    (window as any)._polyPoints = [];
    setPolyPoints([]);
  }, [drawMode]);

  // Helper to calculate area
  const calculatePolygonAreaKm2 = (pts: number[][]): number => {
    if (pts.length < 3) return 0.0;
    const R = 6371.0;
    let totalRad = 0.0;
    for (let i = 0; i < pts.length - 1; i++) {
      const lon1 = (pts[i][0] * Math.PI) / 180.0;
      const lat1 = (pts[i][1] * Math.PI) / 180.0;
      const lon2 = (pts[i + 1][0] * Math.PI) / 180.0;
      const lat2 = (pts[i + 1][1] * Math.PI) / 180.0;
      totalRad += (lon2 - lon1) * (2 + Math.sin(lat1) + Math.sin(lat2));
    }
    return Math.round(Math.abs((totalRad * (R * R)) / 2.0) * 100) / 100;
  };

  const handleResetAOI = () => {
    const lat = center[0];
    const lon = center[1];
    const dLat = 0.04;
    const dLon = 0.05;
    const defaultCoords = [
      [lon - dLon, lat - dLat],
      [lon + dLon, lat - dLat],
      [lon + dLon, lat + dLat],
      [lon - dLon, lat + dLat],
      [lon - dLon, lat - dLat],
    ];
    const newAoi: CopernicusAOI = {
      type: "Polygon",
      coordinates: [defaultCoords],
      bbox: [lon - dLon, lat - dLat, lon + dLon, lat + dLat],
    };
    if (onAOIChange) {
      onAOIChange(newAoi, calculatePolygonAreaKm2(defaultCoords));
    }
    setDrawMode("none");
  };

  const handleDrawRectangleBox = (sizeKm: number = 3) => {
    const lat = center[0];
    const lon = center[1];
    const dLat = (sizeKm / 111.0) / 2.0;
    const dLon = (sizeKm / (111.0 * Math.cos((lat * Math.PI) / 180.0))) / 2.0;

    const coords = [
      [lon - dLon, lat - dLat],
      [lon + dLon, lat - dLat],
      [lon + dLon, lat + dLat],
      [lon - dLon, lat + dLat],
      [lon - dLon, lat - dLat],
    ];
    const newAoi: CopernicusAOI = {
      type: "Polygon",
      coordinates: [coords],
      bbox: [lon - dLon, lat - dLat, lon + dLon, lat + dLat],
    };
    if (onAOIChange) {
      onAOIChange(newAoi, calculatePolygonAreaKm2(coords));
    }
    setDrawMode("none");
  };

  return (
    <div
      className={`relative w-full rounded-xl border border-slate-800 overflow-hidden bg-slate-950 shadow-2xl transition-all ${
        isFullscreen ? "fixed inset-0 z-50 rounded-none h-screen" : "h-[460px]"
      }`}
    >
      {/* Map Canvas */}
      <div ref={mapContainerRef} className="w-full h-full z-0" />

      {/* Top Left Tool Controls */}
      <div className="absolute top-3 left-3 z-10 flex flex-col gap-2">
        {/* Layer Switcher & Draw Mode Toolbar */}
        <div className="flex items-center space-x-1.5 p-1 bg-slate-900/90 backdrop-blur-md rounded-lg border border-slate-700/80 shadow-xl">
          <button
            onClick={() => setBaseMap(baseMap === "satellite" ? "osm" : "satellite")}
            className="flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors text-slate-300 hover:text-emerald-400 hover:bg-slate-800"
            title="Toggle Base Map Tile Layer"
          >
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span>{baseMap === "satellite" ? "Satellite" : "OpenStreetMap"}</span>
          </button>

          <div className="w-px h-4 bg-slate-700" />

          <button
            onClick={() => handleDrawRectangleBox(4)}
            className="flex items-center space-x-1 px-2 py-1 rounded text-xs font-mono transition-colors text-slate-300 hover:text-amber-400 hover:bg-slate-800"
            title="Set 4km AOI Box"
          >
            <Square className="w-3.5 h-3.5 text-emerald-400" />
            <span>AOI Box</span>
          </button>

          <button
            onClick={() => setDrawMode(drawMode === "polygon" ? "none" : "polygon")}
            className={`flex items-center space-x-1 px-2 py-1 rounded text-xs font-mono transition-colors ${
              drawMode === "polygon"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                : "text-slate-300 hover:text-amber-400 hover:bg-slate-800"
            }`}
            title="Click 4+ points on map to draw custom polygon"
          >
            <Compass className="w-3.5 h-3.5 text-amber-400" />
            <span>{drawMode === "polygon" ? `Points: ${polyPoints.length}` : "Draw Poly"}</span>
          </button>

          <button
            onClick={handleResetAOI}
            className="p-1 text-slate-400 hover:text-rose-400 hover:bg-slate-800 rounded transition-colors"
            title="Reset AOI to Center"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Informational Banner if Drawing Polygon */}
        {drawMode === "polygon" && (
          <div className="px-3 py-1.5 bg-amber-950/90 border border-amber-500/40 text-amber-300 text-[11px] font-mono rounded-lg shadow-lg animate-pulse">
            Click points on map to trace AOI polygon ({polyPoints.length}/4 points)
          </div>
        )}
      </div>

      {/* Top Right Controls */}
      <div className="absolute top-3 right-12 z-10 flex items-center space-x-2">
        <button
          onClick={() => setIsFullscreen(!isFullscreen)}
          className="p-1.5 bg-slate-900/90 backdrop-blur-md rounded-lg border border-slate-700/80 text-slate-300 hover:text-cyan-400 hover:bg-slate-800 shadow-xl transition-colors"
          title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Map"}
        >
          {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>
      </div>

      {/* Bottom Floating Stats & Opacity Bar */}
      <div className="absolute bottom-3 left-3 right-3 z-10 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* AOI Details Pill */}
        <div className="flex items-center space-x-2.5 px-3 py-1.5 bg-slate-900/90 backdrop-blur-md rounded-lg border border-slate-700/80 text-xs font-mono text-slate-200 pointer-events-auto shadow-lg">
          <div className="flex items-center space-x-1.5 text-emerald-400 font-bold">
            <MapPin className="w-3.5 h-3.5" />
            <span>AOI: {areaKm2 > 0 ? areaKm2 : "4.82"} km²</span>
          </div>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400 text-[11px]">
            [{center[0].toFixed(3)}°N, {center[1].toFixed(3)}°E]
          </span>
          <span className="text-slate-600">|</span>
          <span className="text-cyan-400 text-[11px]">Native: 10m/px</span>
        </div>

        {/* Overlay Opacity Slider if Satellite Raster Loaded */}
        {overlayImageUrl && (
          <div className="flex items-center space-x-2 px-3 py-1.5 bg-slate-900/90 backdrop-blur-md rounded-lg border border-slate-700/80 text-xs font-mono text-slate-200 pointer-events-auto shadow-lg ml-auto">
            <Sliders className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[11px] text-slate-400">RASTER OPACITY:</span>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(Number(e.target.value))}
              className="w-20 h-1 bg-slate-700 rounded appearance-none cursor-pointer accent-amber-400"
            />
            <span className="text-[11px] text-amber-300">{Math.round(overlayOpacity * 100)}%</span>
          </div>
        )}
      </div>

      {/* Satellite Spectral Legend */}
      {overlayLegend && Object.keys(overlayLegend).length > 0 && (
        <div className="absolute top-16 right-3 z-10 p-2.5 bg-slate-900/90 backdrop-blur-md rounded-lg border border-slate-700/80 text-[11px] font-mono text-slate-300 shadow-xl max-w-[240px] space-y-1.5">
          <div className="font-bold text-slate-400 uppercase tracking-wider text-[10px] pb-1 border-b border-slate-800 flex items-center justify-between">
            <span>{overlayTitle || "SPECTRAL INDEX"}</span>
            <span className="text-emerald-400">10m</span>
          </div>
          <div className="space-y-1">
            {Object.entries(overlayLegend).map(([label, color]) => (
              <div key={label} className="flex items-center space-x-2">
                <span className="w-3 h-3 rounded-sm border border-black/40 flex-shrink-0" style={{ backgroundColor: color }} />
                <span className="truncate text-[10px]">{label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
