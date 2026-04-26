"""Exporter: render full session results as Markdown or PDF.

Reads the v2 session state keys (``match_results``, ``validated_gaps``, etc.)
and produces a single flat document for download.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any


class Exporter:
    """Render session data as Markdown or PDF bytes."""

    def to_markdown(self, session_data: dict[str, Any]) -> str:
        """Render full results as a Markdown string."""
        lines: list[str] = []
        lines.append("# Skill Assessment & Learning Plan")
        lines.append("")
        lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
        lines.append("")

        match_results = session_data.get("match_results") or {}
        gap_band = session_data.get("gap_band", "")
        validated = session_data.get("validated_gaps") or {}
        jd_requirements = session_data.get("jd_requirements") or {}
        skill_matches = session_data.get("skill_matches") or {}
        plan = session_data.get("learning_plan") or {}

        # ---- Overall match ----
        pct = match_results.get("overall_match_percentage")
        if pct is not None:
            lines.append("## Overall Match")
            lines.append("")
            lines.append(f"- **Resume match:** {pct:.1f}%")
            final_pct = validated.get("final_match_percentage")
            if isinstance(final_pct, (int, float)):
                lines.append(f"- **Final match:** {final_pct:.1f}%")
            if gap_band:
                lines.append(f"- **Gap classification:** {gap_band}")
            needing = len(validated.get("skills_needing_attention", []))
            if validated:
                lines.append(f"- **Skills needing attention:** {needing}")
            lines.append("")

        # ---- JD requirements ----
        required_skills = jd_requirements.get("required_skills", [])
        if required_skills:
            lines.append("## JD Requirements")
            lines.append("")
            lines.append("| Skill | Required Level | Importance | Type |")
            lines.append("|---|---|---|---|")
            for s in required_skills:
                lines.append(
                    f"| {s.get('skill_name', '')} "
                    f"| {s.get('required_level', '')}/5 "
                    f"| {s.get('importance', '')} "
                    f"| {s.get('requirement_type', '')} |"
                )
            lines.append("")

        # ---- Resume skill matches ----
        matches = skill_matches.get("skill_matches", [])
        if matches:
            lines.append("## Resume Skill Matches")
            lines.append("")
            lines.append("| Skill | Matched | Match Type | Evidence | Confidence |")
            lines.append("|---|---|---|---|---|")
            for m in matches:
                lines.append(
                    f"| {m.get('skill_name', '')} "
                    f"| {'✓' if m.get('matched') else '✗'} "
                    f"| {m.get('match_type', '')} "
                    f"| {m.get('evidence_strength', '')} "
                    f"| {m.get('confidence', '')} |"
                )
            lines.append("")

        # ---- Weighted match breakdown ----
        breakdown = match_results.get("per_skill_breakdown", [])
        if breakdown:
            lines.append("## Weighted Match Breakdown")
            lines.append("")
            lines.append("| Skill | Importance | Score | Weight | Weighted |")
            lines.append("|---|---|---|---|---|")
            for b in breakdown:
                lines.append(
                    f"| {b.get('skill_name', '')} "
                    f"| {b.get('importance', '')} "
                    f"| {b.get('score', '')} "
                    f"| {b.get('weight', '')} "
                    f"| {b.get('weighted_score', '')} |"
                )
            lines.append("")

        # ---- Validated gap analysis ----
        revised_skills = validated.get("revised_skills", [])
        if revised_skills:
            lines.append("## Validated Gap Analysis")
            lines.append("")
            lines.append(
                "| Skill | Required | Current | Gap | Status | Turns |"
            )
            lines.append("|---|---|---|---|---|---|")
            for s in revised_skills:
                lines.append(
                    f"| {s.get('skill_name', '')} "
                    f"| {s.get('required_level', '')}/5 "
                    f"| {s.get('validated_current_level', '')}/5 "
                    f"| {s.get('gap_size', '')} "
                    f"| {s.get('gap_status', '')} "
                    f"| {s.get('assessment_turns_count', 0)} |"
                )
            lines.append("")

        # ---- Learning plan ----
        plan_items = plan.get("plan_items", [])
        adjacent = plan.get("adjacent_skills", [])
        if plan_items or adjacent:
            lines.append("## Personalised Learning Plan")
            lines.append("")
            total = plan.get("total_estimated_hours", 0)
            lines.append(f"**Total estimated time:** {total} hours")
            lines.append("")

            for item in plan_items:
                gap_type = item.get("gap_type", "").replace("_", " ").title()
                lines.append(
                    f"### {item.get('skill_name', '')} ({gap_type})"
                )
                lines.append("")
                lines.append(
                    f"- Current: {item.get('current_level', '')}/5 "
                    f"→ Target: {item.get('target_level', '')}/5"
                )
                lines.append(
                    f"- Priority: {item.get('priority', '').title()}"
                )
                lines.append(
                    f"- Estimated time: {item.get('estimated_hours', '')} hours"
                )
                if item.get("why_this_matters"):
                    lines.append(f"- Why it matters: {item['why_this_matters']}")
                if item.get("learning_objective"):
                    lines.append(f"- Objective: {item['learning_objective']}")
                deps = item.get("dependency_skills", [])
                if deps:
                    lines.append(f"- Dependencies: {', '.join(deps)}")
                if item.get("practice_task"):
                    lines.append(f"- Practice task: {item['practice_task']}")
                if item.get("checkpoint"):
                    lines.append(f"- Checkpoint: {item['checkpoint']}")
                lines.append("")
                lines.append("**Resources:**")
                for r in item.get("resources", []):
                    title = r.get("title", "")
                    rtype = r.get("type", "")
                    level = r.get("level", "")
                    url = r.get("url", "")
                    if url:
                        lines.append(f"- [{title}]({url}) — _{rtype}, {level}_")
                    else:
                        lines.append(f"- **{title}** — _{rtype}, {level}_")
                lines.append("")

            if adjacent:
                lines.append("### Adjacent Skills Worth Exploring")
                lines.append("")
                for item in adjacent:
                    lines.append(f"#### {item.get('skill_name', '')}")
                    lines.append("")
                    if item.get("why_this_matters"):
                        lines.append(f"- {item['why_this_matters']}")
                    lines.append(
                        f"- Estimated time: {item.get('estimated_hours', '')} hours"
                    )
                    lines.append("")
                    lines.append("**Resources:**")
                    for r in item.get("resources", []):
                        title = r.get("title", "")
                        rtype = r.get("type", "")
                        level = r.get("level", "")
                        url = r.get("url", "")
                        if url:
                            lines.append(f"- [{title}]({url}) — _{rtype}, {level}_")
                        else:
                            lines.append(f"- **{title}** — _{rtype}, {level}_")
                    lines.append("")

        # ---- Scoring rubric footer ----
        lines.append("---")
        lines.append("")
        lines.append("### Proficiency Scale")
        lines.append("")
        lines.append("- **1** — Exposure helpful")
        lines.append("- **2** — Basic familiarity")
        lines.append("- **3** — Working knowledge")
        lines.append("- **4** — Strong hands-on proficiency")
        lines.append("- **5** — Expert / lead-level mastery")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _soft_break_long_tokens(text: str, max_token_len: int = 60) -> str:
        """Insert spaces inside unbreakable tokens so fpdf can wrap them.

        fpdf raises "Not enough horizontal space to render a single character"
        when a single token is wider than the page's usable width. Break
        tokens longer than ``max_token_len`` chars at that boundary.
        """
        out_tokens: list[str] = []
        for tok in text.split(" "):
            if len(tok) <= max_token_len:
                out_tokens.append(tok)
                continue
            # Slice long tokens (URLs, ids) into fixed-width chunks separated by spaces.
            chunks = [tok[i : i + max_token_len] for i in range(0, len(tok), max_token_len)]
            out_tokens.append(" ".join(chunks))
        return " ".join(out_tokens)

    def _safe_multi_cell(self, pdf: Any, height: float, text: str) -> None:
        """Render a line with fpdf, never letting a single bad line abort export."""
        safe = self._soft_break_long_tokens(text).encode("latin-1", "replace").decode("latin-1")
        try:
            pdf.multi_cell(0, height, safe)
        except Exception:
            # Fall back to a truncated line; if that also fails, skip it.
            try:
                pdf.multi_cell(0, height, safe[:200] + "...")
            except Exception:
                pdf.ln(height)

    def to_pdf(self, session_data: dict[str, Any]) -> bytes:
        """Render as PDF bytes. Falls back to Markdown-as-text if fpdf2 missing."""
        try:
            from fpdf import FPDF
        except ImportError:
            return self.to_markdown(session_data).encode("utf-8")

        md = self.to_markdown(session_data)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_margins(left=10, top=10, right=10)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=10)

        for raw_line in md.splitlines():
            line = raw_line.rstrip()
            if not line:
                pdf.ln(3)
                continue

            if line.startswith("# "):
                pdf.set_font("Helvetica", style="B", size=16)
                self._safe_multi_cell(pdf, 8, line[2:])
                pdf.set_font("Helvetica", size=10)
            elif line.startswith("## "):
                pdf.set_font("Helvetica", style="B", size=13)
                self._safe_multi_cell(pdf, 7, line[3:])
                pdf.set_font("Helvetica", size=10)
            elif line.startswith("### "):
                pdf.set_font("Helvetica", style="B", size=11)
                self._safe_multi_cell(pdf, 6, line[4:])
                pdf.set_font("Helvetica", size=10)
            elif line.startswith("#### "):
                pdf.set_font("Helvetica", style="B", size=10)
                self._safe_multi_cell(pdf, 5, line[5:])
                pdf.set_font("Helvetica", size=10)
            else:
                self._safe_multi_cell(pdf, 5, line)

        out = io.BytesIO()
        pdf_bytes = pdf.output(dest="S")
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin-1")
        out.write(bytes(pdf_bytes))
        return out.getvalue()
