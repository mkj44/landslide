"""Temporal persistence and cooldown smoothing for operational landslide early warning."""

from typing import List, Dict, Any, Optional
import numpy as np

TIER_ORDER = ["Low", "Moderate", "High", "Critical"]
TIER_TO_INDEX = {tier: i for i, tier in enumerate(TIER_ORDER)}


def apply_persistence_and_cooldown(
    raw_risk_levels: List[str],
    probabilities: List[float],
    required_consecutive_days: int = 2,
    cooldown_days: int = 3,
    emergency_probability_override: float = 0.85,
) -> List[Dict[str, Any]]:
    """Applies temporal persistence and post-event hydrological cooldown rules.

    Operational Logic:
      1. Persistence (Escalation): Moving to 'High' or 'Critical' requires either:
         - consecutive days >= required_consecutive_days meeting that threshold, OR
         - instantaneous extreme rainfall risk (probability >= emergency_probability_override).
      2. Cooldown (De-escalation): After an elevated alert ('High' or 'Critical'),
         soil saturation and pore-water pressure require 2-3 days of drainage. The alert
         decays at most one risk tier per day rather than dropping immediately to 'Low'.
    """
    n = len(raw_risk_levels)
    if n == 0:
        return []

    smoothed_records: List[Dict[str, Any]] = []
    consecutive_high_count = 0
    cooldown_remaining = 0
    current_tier_idx = 0

    for i in range(n):
        raw_level = raw_risk_levels[i]
        prob = probabilities[i]
        raw_idx = TIER_TO_INDEX.get(raw_level, 0)

        # 1. Escalation check
        if raw_idx >= 2:  # High or Critical
            consecutive_high_count += 1
        else:
            consecutive_high_count = 0

        # Can we escalate?
        can_escalate = (
            (consecutive_high_count >= required_consecutive_days)
            or (prob >= emergency_probability_override)
            or (raw_idx < 2) # Low/Moderate escalate naturally
        )

        if can_escalate:
            effective_raw_idx = raw_idx
        else:
            # Suppress premature single-day spike to Moderate
            effective_raw_idx = min(raw_idx, 1)

        # 2. Cooldown check
        if effective_raw_idx >= 2:
            cooldown_remaining = cooldown_days
            final_idx = effective_raw_idx
        elif cooldown_remaining > 0:
            # Decay at most 1 level below previous day
            final_idx = max(effective_raw_idx, current_tier_idx - 1)
            cooldown_remaining -= 1
        else:
            final_idx = effective_raw_idx

        final_idx = int(np.clip(final_idx, 0, len(TIER_ORDER) - 1))
        current_tier_idx = final_idx
        final_level = TIER_ORDER[final_idx]

        status = "normal"
        if final_idx > effective_raw_idx:
            status = "cooldown_active"
        elif raw_idx > effective_raw_idx:
            status = "persistence_suppressed"

        smoothed_records.append({
            "step": i,
            "raw_risk_level": raw_level,
            "effective_risk_level": final_level,
            "probability": round(prob, 4),
            "cooldown_remaining": cooldown_remaining,
            "status": status,
        })

    return smoothed_records
