"""Deterministic scoring and gap-analysis helpers for the v2 pipeline.

Pure Python math only: no LLM calls. These functions take the structured
outputs from the LLM steps (JD requirements, semantic skill matches,
assessment results) and produce the overall match percentage and the
validated gap analysis.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


# Weights applied per JD importance bucket.
IMPORTANCE_WEIGHTS = {"high": 1.0, "medium": 0.7, "low": 0.4}

# Base scores for each match-type returned by the semantic skill match step.
MATCH_SCORES = {
    "exact_match": 1.0,
    "semantic_match": 0.9,
    "adjacent_match": 0.55,
    "no_match": 0.0,
}

# Multipliers reflecting how strong the resume evidence is for a matched skill.
EVIDENCE_MODIFIER = {"strong": 1.0, "moderate": 0.85, "weak": 0.65, "none": 0.0}

# Multipliers reflecting the LLM's self-reported confidence in the match.
CONFIDENCE_MODIFIER = {"high": 1.0, "medium": 0.9, "low": 0.75}


def _years_modifier(
    years_required: float | int | None,
    years_experience: float | int | None,
) -> tuple[float, float | None]:
    """Return (modifier, ratio) based on the years-gap buckets.

    Rules (see design.md § "Years-Gap Modifier in STEP2 Scoring"):
      - years_required is None / 0 / negative -> neutral (1.0, None).
      - years_experience is None / negative -> neutral (1.0, None).
        (evidence_strength already reflects absence of evidence.)
      - Otherwise ratio = min(1.0, years_experience / years_required), then:
            ratio >= 1.0     -> 1.0
            0.66 <= r < 1.0  -> 0.9
            0.33 <= r < 0.66 -> 0.75
            0 <= r < 0.33    -> 0.6
    """
    try:
        req = float(years_required) if years_required is not None else 0.0
    except (TypeError, ValueError):
        req = 0.0
    if req <= 0:
        return 1.0, None

    if years_experience is None:
        return 1.0, None
    try:
        exp = float(years_experience)
    except (TypeError, ValueError):
        return 1.0, None
    if exp < 0:
        return 1.0, None

    ratio = max(0.0, min(1.0, exp / req))

    if ratio >= 1.0:
        modifier = 1.0
    elif ratio >= 0.66:
        modifier = 0.9
    elif ratio >= 0.33:
        modifier = 0.75
    else:
        modifier = 0.6

    return modifier, round(ratio, 3)


def compute_match_percentage(
    jd_requirements: dict[str, Any],
    skill_matches: dict[str, Any],
) -> dict[str, Any]:
    """Compute the weighted overall match percentage and per-skill breakdown.

    Combines match_type, evidence_strength, confidence and the years-gap
    modifier into a per-skill score, then weights each entry by JD
    importance and requirement_type.
    """
    # Index matches by lowercased skill name for O(1) lookup.
    match_map = {
        item["skill_name"].strip().lower(): item
        for item in skill_matches.get("skill_matches", [])
    }

    per_skill_breakdown: list[dict[str, Any]] = []
    total_weight = 0.0
    total_weighted_score = 0.0

    for req in jd_requirements.get("required_skills", []):
        skill_name = req["skill_name"]
        key = skill_name.strip().lower()
        item = match_map.get(key, {})

        # Base weight from importance; "required" items get a +0.2 bump.
        weight = IMPORTANCE_WEIGHTS.get(req.get("importance", "medium"), 0.7)
        if req.get("requirement_type") == "required":
            weight += 0.2

        base = MATCH_SCORES.get(item.get("match_type", "no_match"), 0.0)
        ev_mod = EVIDENCE_MODIFIER.get(item.get("evidence_strength", "none"), 0.0)
        conf_mod = CONFIDENCE_MODIFIER.get(item.get("confidence", "low"), 0.75)

        years_required = req.get("years_required")
        years_experience = item.get("years_experience")
        years_mod, years_ratio = _years_modifier(years_required, years_experience)

        score = round(base * ev_mod * conf_mod * years_mod, 3)
        weighted_score = round(score * weight, 3)

        per_skill_breakdown.append(
            {
                "skill_name": skill_name,
                "requirement_type": req.get("requirement_type"),
                "importance": req.get("importance"),
                "required_level": req.get("required_level"),
                "match_type": item.get("match_type", "no_match"),
                "matched": item.get("matched", False),
                "evidence_strength": item.get("evidence_strength", "none"),
                "confidence": item.get("confidence", "low"),
                "years_required": years_required,
                "years_experience": years_experience,
                "years_ratio": years_ratio,
                "years_modifier": years_mod,
                "score": score,
                "weight": round(weight, 3),
                "weighted_score": weighted_score,
                "matched_resume_terms": item.get("matched_resume_terms", []),
                "evidence_spans": item.get("evidence_spans", []),
            }
        )

        total_weight += weight
        total_weighted_score += weighted_score

    overall = (
        0.0
        if total_weight == 0
        else round((total_weighted_score / total_weight) * 100, 2)
    )
    return {
        "overall_match_percentage": overall,
        "per_skill_breakdown": per_skill_breakdown,
    }


def gap_classification(match_percentage: float) -> str:
    """Return the gap band: Low (<50), Medium (50-75), High (>75)."""
    if not 0 <= match_percentage <= 100:
        raise ValueError(f"match_percentage must be in 0-100, got {match_percentage}")
    if match_percentage < 50:
        return "Low"
    if match_percentage <= 75:
        return "Medium"
    return "High"


def _resume_baseline_level(match_type: str, evidence_strength: str) -> int:
    """Estimate a 1-5 current-level floor from resume evidence alone."""
    if match_type == "exact_match" and evidence_strength == "strong":
        return 3
    if match_type in {"exact_match", "semantic_match"} and evidence_strength in {"moderate", "strong"}:
        return 2
    if match_type == "adjacent_match" and evidence_strength in {"weak", "moderate", "strong"}:
        return 1
    return 1


def build_validated_gaps(
    jd_requirements: dict[str, Any],
    skill_matches: dict[str, Any],
    assessment_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine resume evidence with assessment results into validated skill gaps."""
    # Index both inputs by lowercased skill name.
    match_map = {
        item["skill_name"].strip().lower(): item
        for item in skill_matches.get("skill_matches", [])
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in assessment_results:
        grouped[item["skill_name"].strip().lower()].append(item)

    revised_skills: list[dict[str, Any]] = []
    final_total_weight = 0.0
    final_total_weighted_score = 0.0

    for req in jd_requirements.get("required_skills", []):
        skill_name = req["skill_name"]
        key = skill_name.strip().lower()
        match = match_map.get(key, {})
        turns = grouped.get(key, [])

        baseline = _resume_baseline_level(
            match.get("match_type", "no_match"),
            match.get("evidence_strength", "none"),
        )

        # Blend baseline with assessment turns when available.
        if turns:
            avg_level = sum(int(t.get("provisional_level", 1)) for t in turns) / len(turns)
            avg_score = sum(float(t.get("overall_score", 1.0)) for t in turns) / len(turns)

            if avg_score >= 4.0:
                validated_current_level = max(baseline, round(avg_level))
            elif avg_score >= 3.0:
                validated_current_level = max(1, round((baseline + avg_level) / 2))
            else:
                validated_current_level = min(baseline, round(avg_level))
        else:
            validated_current_level = baseline

        validated_current_level = max(1, min(5, int(validated_current_level)))
        required_level = req.get("required_level") or 3
        gap_size = max(0, required_level - validated_current_level)

        if gap_size == 0:
            gap_status = "met"
        elif gap_size == 1:
            gap_status = "partial_gap"
        else:
            gap_status = "critical_gap"

        # ---- Final match % per-skill contribution ----
        # level_ratio clamps to [0, 1] so over-qualification cannot exceed 100%.
        level_ratio = max(0.0, min(1.0, validated_current_level / float(required_level)))

        final_weight = IMPORTANCE_WEIGHTS.get(req.get("importance", "medium"), 0.7)
        if req.get("requirement_type") == "required":
            final_weight += 0.2
        final_weight = round(final_weight, 3)

        final_weighted_score = round(level_ratio * final_weight, 3)

        final_total_weight += final_weight
        final_total_weighted_score += final_weighted_score

        revised_skills.append(
            {
                "skill_name": skill_name,
                "requirement_type": req.get("requirement_type"),
                "importance": req.get("importance"),
                "required_level": required_level,
                "validated_current_level": validated_current_level,
                "gap_size": gap_size,
                "gap_status": gap_status,
                "match_type": match.get("match_type", "no_match"),
                "resume_evidence_strength": match.get("evidence_strength", "none"),
                "assessment_turns_count": len(turns),
                "level_ratio": round(level_ratio, 3),
                "final_weight": final_weight,
                "final_weighted_score": final_weighted_score,
                "assessment_summary": [
                    {
                        "overall_score": t.get("overall_score"),
                        "provisional_level": t.get("provisional_level"),
                        "confidence": t.get("confidence"),
                        "strengths": t.get("strengths", []),
                        "weaknesses": t.get("weaknesses", []),
                    }
                    for t in turns
                ],
            }
        )

    # Largest gap first; break ties by importance=high.
    revised_skills.sort(
        key=lambda s: (s["gap_size"], 1 if s["importance"] == "high" else 0),
        reverse=True,
    )

    final_match_percentage = (
        0.0
        if final_total_weight == 0
        else round((final_total_weighted_score / final_total_weight) * 100, 2)
    )

    return {
        "revised_skills": revised_skills,
        "skills_needing_attention": [s for s in revised_skills if s["gap_size"] > 0],
        "final_match_percentage": final_match_percentage,
    }
