"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Search,
  MapPin,
  Sparkles,
  Layers,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Compass,
  Sliders,
  TrendingUp,
  Brain,
  Filter,
  Eye,
  Maximize2,
  RefreshCw,
  Info,
  ShieldAlert,
  ArrowRight,
  Send,
  Building,
  Trees,
  Droplets,
  HelpCircle
} from "lucide-react";
import SatelliteMap from "./SatelliteMap";
import {
  executeGeospatialSearch,
  parseGeospatialQuery,
  executeSearchFollowup
} from "../lib/api";
import {
  GeospatialSearchResponseData,
  CandidateRegionData,
  CopernicusAOI,
  GroundingBox
} from "../lib/types";

const SUGGESTED_QUERIES = [
  "Find potentially vacant land around Perundurai",
  "Find large empty lands within 10 km of Perundurai",
  "Show water bodies around Perundurai",
  "Find areas with poor vegetation health near Erode",
  "Show vegetation loss around Perundurai",
  "Find newly developed areas around Perundurai",
  "Find areas that changed from vegetation to built-up land around Perundurai between 2022 and 2026",
  "Find the best land to buy in Perundurai", // Demonstrates graceful handling of out-of-scope queries
];

export default function GeospatialSearch() {
  const [searchQuery, setSearchQuery] = useState<string>("Find potentially vacant land around Perundurai");
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [searchResult, setSearchResult] = useState<GeospatialSearchResponseData | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [activeRankCriteria, setActiveRankCriteria] = useState<string>("area_desc");
  const [candidateList, setCandidateList] = useState<CandidateRegionData[]>([]);

  // Follow-up state
  const [followupPrompt, setFollowupPrompt] = useState<string>("");
  const [isFollowupLoading, setIsFollowupLoading] = useState<boolean>(false);
  const [chatHistory, setChatHistory] = useState<Array<{ sender: "user" | "ai"; text: string }>>([]);

  // Observation Preview Modal
  const [previewObs, setPreviewObs] = useState<any | null>(null);

  // Map state
  const [mapCenter, setMapCenter] = useState<[number, number]>([11.2750, 77.5833]);
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
  const [aoiAreaKm2, setAoiAreaKm2] = useState<number>(5.42);

  // Run initial search on mount
  useEffect(() => {
    handleExecuteSearch("Find potentially vacant land around Perundurai");
  }, []);

  const handleExecuteSearch = async (queryText: string) => {
    if (!queryText.trim()) return;
    setIsSearching(true);
    setSearchQuery(queryText);
    setSelectedCandidateId(null);

    try {
      const res: GeospatialSearchResponseData = await executeGeospatialSearch({
        query: queryText,
      });

      setSearchResult(res);
      setCandidateList(res.candidates || []);
      setMapCenter(res.center_coordinates || [11.2750, 77.5833]);
      setAoiAreaKm2(res.total_aoi_area_km2 || 5.0);

      if (res.search_aoi_bbox) {
        const [minLon, minLat, maxLon, maxLat] = res.search_aoi_bbox;
        setAoi({
          type: "Polygon",
          coordinates: [
            [
              [minLon, minLat],
              [maxLon, minLat],
              [maxLon, maxLat],
              [minLon, maxLat],
              [minLon, minLat],
            ],
          ],
          bbox: res.search_aoi_bbox,
        });
      }

      setChatHistory([
        { sender: "user", text: queryText },
        { sender: "ai", text: res.summary_explanation },
      ]);
    } catch (err: any) {
      console.error("Search failed:", err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSortCandidates = (criteria: string) => {
    setActiveRankCriteria(criteria);
    const sorted = [...candidateList];
    if (criteria === "distance_asc") {
      sorted.sort((a, b) => a.distance_km - b.distance_km);
    } else if (criteria === "persistence_desc") {
      sorted.sort((a, b) => b.persistence_ratio - a.persistence_ratio);
    } else if (criteria === "confidence_desc") {
      sorted.sort((a, b) => b.confidence_score - a.confidence_score);
    } else {
      sorted.sort((a, b) => b.area_hectares - a.area_hectares);
    }
    setCandidateList(sorted);
  };

  const handleSendFollowup = async () => {
    if (!followupPrompt.trim() || !searchResult) return;
    const prompt = followupPrompt;
    setFollowupPrompt("");
    setIsFollowupLoading(true);

    setChatHistory((prev) => [...prev, { sender: "user", text: prompt }]);

    try {
      const res = await executeSearchFollowup({
        prompt: prompt,
        candidates: candidateList,
        location: searchResult.location_name,
      });

      setChatHistory((prev) => [...prev, { sender: "ai", text: res.reply }]);
      if (res.filtered_candidates) {
        setCandidateList(res.filtered_candidates);
      }
    } catch (err: any) {
      setChatHistory((prev) => [
        ...prev,
        { sender: "ai", text: `Error processing follow-up: ${err.message}` },
      ]);
    } finally {
      setIsFollowupLoading(false);
    }
  };

  // Convert candidates to map GroundingBox format
  const mapGroundingBoxes: GroundingBox[] = candidateList.map((c) => ({
    id: c.id,
    label: `${c.name} (${c.area_hectares} ha)`,
    box: c.bounding_box,
    confidence: c.confidence_score,
    color: selectedCandidateId === c.id ? "#38bdf8" : c.color,
  }));

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* 1. Header & Geospatial Intelligence Banner */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-cyan-950/40 to-slate-900 border border-cyan-500/30 shadow-2xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2.5">
            <Compass className="w-5 h-5 text-cyan-400 animate-spin" style={{ animationDuration: "12s" }} />
            <h2 className="text-lg font-bold text-slate-100 tracking-tight">
              Natural-Language Geospatial Search & Earth Observation Engine
            </h2>
            <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-[10px] font-mono">
              AI QUERY PLANNER ACTIVE
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Ask complex Earth observation queries in plain English. Translates intent $\rightarrow$ Copernicus Sentinel-2 discovery $\rightarrow$ multi-spectral land intelligence $\rightarrow$ ranked candidate polygons.
          </p>
        </div>

        <div className="flex items-center space-x-2 text-xs font-mono text-slate-300 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
          <Sparkles className="w-4 h-4 text-amber-400" />
          <span>Real-time Sentinel-2 BOA Retrieval</span>
        </div>
      </div>

      {/* 2. Prominent Natural Language Geospatial Search Box */}
      <div className="p-5 rounded-xl bg-slate-900/90 border border-slate-800 shadow-2xl space-y-4">
        <div className="space-y-2">
          <label className="text-xs font-mono text-slate-300 font-bold flex items-center space-x-2">
            <Search className="w-4 h-4 text-cyan-400" />
            <span>ASK ABOUT AN AREA (NATURAL LANGUAGE QUERY):</span>
          </label>

          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleExecuteSearch(searchQuery);
                }}
                placeholder="e.g. Find potentially vacant land around Perundurai..."
                className="w-full pl-4 pr-10 py-3 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 shadow-inner font-sans"
              />
            </div>
            <button
              onClick={() => handleExecuteSearch(searchQuery)}
              disabled={isSearching}
              className="px-6 py-3 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs rounded-xl transition-all flex items-center space-x-2 shadow-lg shadow-cyan-900/30 whitespace-nowrap"
            >
              {isSearching ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
              <span>{isSearching ? "Analyzing Earth Observation..." : "Search Imagery"}</span>
            </button>
          </div>
        </div>

        {/* Suggested Query Chips */}
        <div className="flex items-center space-x-1.5 overflow-x-auto text-xs font-mono pt-1">
          <span className="text-slate-500 text-[11px] uppercase mr-1 flex-shrink-0">SUGGESTIONS:</span>
          {SUGGESTED_QUERIES.map((q) => (
            <button
              key={q}
              onClick={() => handleExecuteSearch(q)}
              className={`px-2.5 py-1 rounded border text-[11px] whitespace-nowrap transition-colors ${
                searchQuery === q
                  ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/50 font-bold"
                  : "bg-slate-950/80 text-slate-400 hover:text-slate-200 border-slate-800"
              }`}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* 3. Legal & Geophysical Classification Disclaimer */}
      <div className="p-3.5 bg-amber-950/30 border border-amber-500/40 rounded-xl text-xs text-amber-200 flex items-start space-x-3 shadow-lg">
        <ShieldAlert className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-0.5 leading-relaxed font-sans">
          <span className="font-bold font-mono uppercase text-amber-300 block">
            GEOPHYSICAL DETECTION NOTICE (POTENTIALLY VACANT / BARE LAND)
          </span>
          <p className="text-[12px] text-amber-200/90">
            Satellite imagery identifies areas that appear bare, non-vegetated, or unbuilt based on multispectral reflection.
            This does <strong>NOT</strong> confirm legal property ownership, boundary demarcations, zoning permissions, or availability for purchase.
          </p>
        </div>
      </div>

      {/* 4. Live Execution Steps Progress Stepper */}
      {searchResult?.execution_steps && searchResult.execution_steps.length > 0 && (
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-2.5">
          <div className="flex items-center justify-between pb-1 border-b border-slate-800 text-xs font-mono">
            <span className="text-slate-400 font-bold">AUTONOMOUS EXECUTION TRACE</span>
            <span className="text-emerald-400">{searchResult.execution_time_ms} ms</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-5 gap-2 text-[11px] font-mono">
            {searchResult.execution_steps.map((st) => (
              <div key={st.step} className="p-2 bg-slate-950 rounded-lg border border-slate-800/80 space-y-1">
                <div className="flex items-center space-x-1.5 text-emerald-400 font-bold">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span className="truncate">{st.name}</span>
                </div>
                <p className="text-[10px] text-slate-400 truncate">{st.detail}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. Main Map & Candidates Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Interactive Map with Detected Polygons */}
        <div className="lg:col-span-2 space-y-3">
          <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
            <div className="flex items-center justify-between pb-1 border-b border-slate-800">
              <div className="flex items-center space-x-2 text-sm font-bold text-slate-200">
                <MapPin className="w-4 h-4 text-cyan-400" />
                <span>Geospatial Map ({searchResult?.location_name || "Perundurai"})</span>
              </div>
              <span className="text-xs font-mono text-slate-400">
                Found {candidateList.length} candidate polygon(s)
              </span>
            </div>

            <SatelliteMap
              center={mapCenter}
              aoi={aoi}
              onAOIChange={(newAoi, area) => {
                setAoi(newAoi);
                setAoiAreaKm2(area);
              }}
              areaKm2={aoiAreaKm2}
              overlayImageUrl={searchResult?.analysis_layer_url}
              overlayLegend={searchResult?.layer_legend}
              overlayTitle={searchResult?.structured_query?.analysis_type?.toUpperCase() || "SPECTRAL ANALYSIS"}
              groundingBoxes={mapGroundingBoxes}
            />
          </div>

          {/* AI Analysis Summary Explanation */}
          {searchResult?.summary_explanation && (
            <div className="p-4 rounded-xl bg-slate-900/90 border border-cyan-500/30 shadow-xl space-y-2">
              <div className="flex items-center space-x-2 text-xs font-mono text-cyan-300 font-bold">
                <Brain className="w-4 h-4 text-cyan-400" />
                <span>AI ANALYSIS REASONING SUMMARY</span>
              </div>
              <p className="text-xs text-slate-200 leading-relaxed font-sans whitespace-pre-line">
                {searchResult.summary_explanation}
              </p>
            </div>
          )}
        </div>

        {/* Right Col: Ranked Candidate Result Cards */}
        <div className="lg:col-span-1 space-y-3 flex flex-col">
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-3 flex-1">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center space-x-2 text-xs font-bold text-slate-200 font-mono">
                <Filter className="w-3.5 h-3.5 text-amber-400" />
                <span>CANDIDATE REGIONS ({candidateList.length})</span>
              </div>

              {/* Ranking Criteria Dropdown */}
              <select
                value={activeRankCriteria}
                onChange={(e) => handleSortCandidates(e.target.value)}
                className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-[11px] font-mono text-slate-300"
              >
                <option value="area_desc">Rank by Area (Largest)</option>
                <option value="distance_asc">Rank by Proximity (Closest)</option>
                <option value="persistence_desc">Rank by Persistence</option>
                <option value="confidence_desc">Rank by Confidence</option>
              </select>
            </div>

            {/* Candidate Cards List */}
            <div className="space-y-2.5 max-h-[520px] overflow-y-auto pr-1">
              {candidateList.map((cand) => {
                const isSelected = selectedCandidateId === cand.id;
                return (
                  <div
                    key={cand.id}
                    onClick={() => setSelectedCandidateId(cand.id)}
                    className={`p-3.5 rounded-xl border transition-all cursor-pointer space-y-2 ${
                      isSelected
                        ? "bg-cyan-950/40 border-cyan-500/60 shadow-lg shadow-cyan-950/20"
                        : "bg-slate-950 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-200 font-mono flex items-center space-x-1.5">
                        <span className="w-4 h-4 rounded-full bg-slate-800 text-slate-300 flex items-center justify-center text-[10px]">
                          #{cand.rank}
                        </span>
                        <span className="truncate max-w-[170px]">{cand.name}</span>
                      </span>
                      <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded font-bold">
                        {cand.area_hectares} ha
                      </span>
                    </div>

                    {/* Metric Badges */}
                    <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono text-slate-400">
                      <div className="bg-slate-900/90 p-1.5 rounded border border-slate-800/80">
                        <span className="block text-slate-500 text-[9px]">DISTANCE:</span>
                        <span className="text-slate-200 font-bold">{cand.distance_km} km</span>
                      </div>
                      <div className="bg-slate-900/90 p-1.5 rounded border border-slate-800/80">
                        <span className="block text-slate-500 text-[9px]">PERSISTENCE:</span>
                        <span className="text-amber-300 font-bold">{cand.temporal_persistence.split(" ")[0]}</span>
                      </div>
                      <div className="bg-slate-900/90 p-1.5 rounded border border-slate-800/80">
                        <span className="block text-slate-500 text-[9px]">VEGETATION:</span>
                        <span className="text-slate-200">{cand.vegetation_score}</span>
                      </div>
                      <div className="bg-slate-900/90 p-1.5 rounded border border-slate-800/80">
                        <span className="block text-slate-500 text-[9px]">CONFIDENCE:</span>
                        <span className="text-emerald-400 font-bold">{cand.confidence_formatted}</span>
                      </div>
                    </div>

                    <p className="text-[11px] text-slate-300 font-sans leading-snug">{cand.description}</p>
                  </div>
                );
              })}

              {candidateList.length === 0 && (
                <p className="text-xs font-mono text-slate-500 text-center py-6">
                  No candidate areas match the current filters.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 6. Conversational Follow-Up Interface */}
      <div className="p-5 rounded-xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <div className="flex items-center space-x-2 text-xs font-bold text-slate-200 font-mono">
            <Brain className="w-4 h-4 text-cyan-400" />
            <span>CONVERSATIONAL FOLLOW-UP QUERY ASSISTANT</span>
          </div>
          <span className="text-[11px] font-mono text-slate-400">Contextual Refinement</span>
        </div>

        {/* Chat message bubbles */}
        {chatHistory.length > 0 && (
          <div className="space-y-2 max-h-[160px] overflow-y-auto pr-1">
            {chatHistory.map((msg, i) => (
              <div
                key={i}
                className={`p-2.5 rounded-lg text-xs font-sans leading-relaxed ${
                  msg.sender === "user"
                    ? "bg-cyan-950/50 border border-cyan-500/30 text-cyan-200 ml-8"
                    : "bg-slate-950 border border-slate-800 text-slate-300 mr-8"
                }`}
              >
                <span className="text-[10px] font-mono uppercase font-bold block mb-0.5 text-slate-400">
                  {msg.sender === "user" ? "You" : "SatQuery Earth AI"}
                </span>
                {msg.text}
              </div>
            ))}
          </div>
        )}

        {/* Follow-up input bar */}
        <div className="flex items-center gap-2 pt-1">
          <input
            type="text"
            value={followupPrompt}
            onChange={(e) => setFollowupPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSendFollowup();
            }}
            placeholder="Ask follow-up (e.g. 'Only show areas larger than 3 hectares', 'Which one is closest?')..."
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
          />
          <button
            onClick={handleSendFollowup}
            disabled={isFollowupLoading || !followupPrompt.trim()}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-slate-950 font-bold text-xs rounded-lg transition-colors flex items-center space-x-1"
          >
            {isFollowupLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
            <span>Ask</span>
          </button>
        </div>
      </div>
    </div>
  );
}
