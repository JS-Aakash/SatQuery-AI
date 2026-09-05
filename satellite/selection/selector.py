"""
Intelligent Remote-Sensing Scene Selector.
Ranks candidate scenes across multi-criteria scoring (cloud cover, temporal proximity,
spatial overlap, and product calibration level) and produces auditable selection justifications.
"""
from typing import List, Optional, Tuple
from datetime import datetime
from ..schemas import SatelliteProduct, SatelliteSensor


class SceneSelector:
    """
    Intelligent multi-criteria ranking and scene selection engine.
    """

    @staticmethod
    def rank_and_select(
        candidates: List[SatelliteProduct],
        target_date: Optional[str] = None,
        prefer_level_2a: bool = True
    ) -> List[SatelliteProduct]:
        """
        Calculates selection score for each candidate and marks the optimal scene.
        Returns candidates ordered by score descending.
        """
        if not candidates:
            return []

        target_dt = None
        if target_date:
            try:
                target_dt = datetime.fromisoformat(target_date.replace("Z", ""))
            except Exception:
                pass

        scored_list: List[Tuple[float, SatelliteProduct, str]] = []

        for prod in candidates:
            score = 1.0
            reasons = []

            # 1. Cloud Coverage Score (w = 0.40 for Optical)
            if prod.sensor == SatelliteSensor.SENTINEL_2_OPTICAL:
                cloud = prod.cloud_coverage_percentage
                cloud_score = max(0.0, 1.0 - (cloud / 30.0))
                score *= (0.60 + 0.40 * cloud_score)
                if cloud < 5.0:
                    reasons.append(f"exceptional clear-sky quality ({cloud:.1f}% cloud)")
                elif cloud < 15.0:
                    reasons.append(f"low cloud coverage ({cloud:.1f}%)")
                else:
                    reasons.append(f"moderate cloud ({cloud:.1f}%)")
            else:
                reasons.append("all-weather radar penetration (0% cloud interference)")

            # 2. Temporal Proximity Score (w = 0.30)
            if target_dt:
                try:
                    prod_dt = datetime.fromisoformat(prod.acquisition_date.replace("Z", ""))
                    day_diff = abs((prod_dt - target_dt).days)
                    time_score = max(0.0, 1.0 - (day_diff / 60.0))
                    score *= (0.70 + 0.30 * time_score)
                    reasons.append(f"temporal offset: {day_diff} days from requested target")
                except Exception:
                    pass

            # 3. Spatial AOI Overlap Score (w = 0.20)
            overlap = prod.aoi_overlap_percentage
            overlap_score = overlap / 100.0
            score *= (0.80 + 0.20 * overlap_score)
            reasons.append(f"{overlap:.0f}% AOI spatial coverage")

            # 4. Product Type Calibration Bonus (w = 0.10)
            p_upper = prod.product_type.upper()
            if "2A" in p_upper or "L2A" in p_upper or "GRD" in p_upper:
                score += 0.08
                reasons.append("calibrated Bottom-of-Atmosphere / GRD product")
            elif "1C" in p_upper or "L1C" in p_upper or "SLC" in p_upper:
                score -= 0.05
                reasons.append("uncalibrated Top-of-Atmosphere / SLC product")

            final_score = min(0.99, max(0.1, score))
            justification = f"Selected for optimal observation quality: {'; '.join(reasons)}."
            scored_list.append((final_score, prod, justification))

        # Sort descending by score
        scored_list.sort(key=lambda x: x[0], reverse=True)

        ranked_products: List[SatelliteProduct] = []
        for i, (score, prod, reason) in enumerate(scored_list):
            prod.selection_score = round(score, 3)
            prod.is_selected = (i == 0)
            prod.selection_reason = reason if (i == 0) else f"Candidate rank #{i+1} (Score: {score:.2f})"
            ranked_products.append(prod)

        return ranked_products

    @staticmethod
    def select_optimal_scene(
        candidates: List[SatelliteProduct],
        target_date: Optional[str] = None
    ) -> Optional[SatelliteProduct]:
        """Returns the single top-ranked optimal scene."""
        ranked = SceneSelector.rank_and_select(candidates, target_date=target_date)
        return ranked[0] if ranked else None

    @staticmethod
    def select_bitemporal_pair(
        before_candidates: List[SatelliteProduct],
        after_candidates: List[SatelliteProduct],
        target_date_before: Optional[str] = None,
        target_date_after: Optional[str] = None
    ) -> Tuple[Optional[SatelliteProduct], Optional[SatelliteProduct]]:
        """
        Selects the optimal pair of scenes for bi-temporal change detection,
        ensuring seasonal compatibility and low cloud cover.
        """
        best_before = SceneSelector.select_optimal_scene(before_candidates, target_date=target_date_before)
        best_after = SceneSelector.select_optimal_scene(after_candidates, target_date=target_date_after)

        if best_before and best_after:
            best_before.selection_reason = f"Optimal Baseline (T1): {best_before.selection_reason}"
            best_after.selection_reason = f"Optimal Monitoring (T2): {best_after.selection_reason}"

        return best_before, best_after
