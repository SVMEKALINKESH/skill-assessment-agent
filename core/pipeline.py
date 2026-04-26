"""Pipeline orchestrator for the v2 agent.

Splits the old single-step skill analysis into discrete steps:

    Step 0  : LLM resume parse.
    Step 1  : LLM JD requirement extraction.
    Step 1B : LLM semantic skill match.
    Step 2  : Code-only match percentage.
    Step 3  : Code-only gap classification.
    Step 4A : LLM assessment question.
    Step 4B : LLM response evaluation.
    Step 5  : Code-only validated gap analysis.
    Step 6  : LLM learning plan.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.agent_prompts import AgentPrompts
from core.llm_provider import LLMProvider
from core.scoring import (
    build_validated_gaps,
    compute_match_percentage,
    gap_classification,
)


# Allow-listed domains for learning resource URLs. Keep in sync with STEP6 prompt.
_ALLOWED_URL_DOMAINS: tuple[str, ...] = (
    "kubernetes.io", "docs.python.org", "developer.mozilla.org", "roadmap.sh",
    "aws.amazon.com", "cloud.google.com", "learn.microsoft.com",
    "coursera.org", "edx.org", "udemy.com", "pluralsight.com",
    "youtube.com", "freecodecamp.org", "fast.ai", "huggingface.co",
    "kaggle.com", "leetcode.com", "hackerrank.com", "github.com",
    "owasp.org", "portswigger.net", "hackthebox.com",
    "docs.docker.com", "nodejs.org", "reactjs.org", "react.dev",
    "spring.io", "pytorch.org", "tensorflow.org",
)


def _is_allowed_url(url: str) -> bool:
    """Validate a URL is https and on the allow-list."""
    if not isinstance(url, str):
        return False
    url = url.strip()
    if not url.startswith(("https://", "http://")):
        return False
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in _ALLOWED_URL_DOMAINS)


def _clamp_hours(value: Any, lo: float = 3.0, hi: float = 30.0) -> float:
    """Coerce ``value`` to float and clamp into ``[lo, hi]``; non-numeric -> ``lo``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = lo
    if v < lo:
        v = lo
    elif v > hi:
        v = hi
    return round(v, 1)


# Canonical fallback URLs for common skills.
#
# Used to back-fill when the LLM omits every URL from a plan_item's
# resources. Keys are lowercase substrings matched against skill_name.
_CANONICAL_FALLBACK_URLS: tuple[tuple[str, str, str], ...] = (
    # (skill keyword, resource title, url)
    ("python", "Python Official Tutorial", "https://docs.python.org/3/tutorial/"),
    ("javascript", "MDN JavaScript Guide", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"),
    ("typescript", "MDN JavaScript Guide", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"),
    ("react", "React Official Docs – Learn", "https://react.dev/learn"),
    ("node", "Node.js Learn", "https://nodejs.org/en/learn"),
    ("docker", "Docker Get Started", "https://docs.docker.com/get-started/"),
    ("kubernetes", "Kubernetes Tutorials", "https://kubernetes.io/docs/tutorials/kubernetes-basics/"),
    ("aws", "AWS Hands-on Tutorials", "https://aws.amazon.com/getting-started/hands-on/"),
    ("gcp", "Google Cloud Docs", "https://cloud.google.com/docs"),
    ("azure", "Microsoft Learn", "https://learn.microsoft.com/en-us/training/"),
    ("sql", "freeCodeCamp SQL", "https://www.freecodecamp.org/news/tag/sql/"),
    ("system design", "System Design Primer", "https://github.com/donnemartin/system-design-primer"),
    ("data structures", "LeetCode Study Plans", "https://leetcode.com/studyplan/"),
    ("algorithm", "LeetCode Study Plans", "https://leetcode.com/studyplan/"),
    ("machine learning", "Kaggle Learn", "https://www.kaggle.com/learn"),
    ("deep learning", "Hugging Face Learn", "https://huggingface.co/learn"),
    ("owasp", "OWASP Top Ten", "https://owasp.org/www-project-top-ten/"),
    ("security", "PortSwigger Web Security Academy", "https://portswigger.net/web-security"),
    ("git", "Git Documentation", "https://github.com/git-guides"),
    ("microservice", "roadmap.sh Backend", "https://roadmap.sh/backend"),
)


def _fallback_url_for_skill(skill_name: str) -> tuple[str, str] | None:
    """Return ``(title, url)`` for the first keyword match, or ``None``."""
    name = (skill_name or "").strip().lower()
    if not name:
        return None
    for keyword, title, url in _CANONICAL_FALLBACK_URLS:
        if keyword in name:
            return title, url
    return None


class PipelineError(Exception):
    """Raised for recoverable errors in the pipeline."""


@dataclass
class AssessmentTurn:
    """One Q/A turn captured during the assessment phase."""

    skill_name: str
    question: str
    response: str
    overall_score: float
    provisional_level: int
    is_relevant: bool
    confidence: str
    dimension_scores: dict[str, int]
    should_ask_follow_up: bool
    follow_up_focus: str
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str


class Pipeline:
    """Drive the v2 agent through its per-step LLM calls."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    # ---- Step 0: Resume parse (LLM) ----

    def step0_parse_resume(self, resume_text: str) -> dict[str, Any]:
        """Parse a resume into structured JSON."""
        self._validate_text(resume_text, "Resume")
        prompt = AgentPrompts.resume_parse(resume_text)
        result = self._call_llm_for_json(prompt, step_name="Step 0 (Resume Parse)")

        for key in ["candidate_profile", "experience", "skills"]:
            if key not in result:
                raise PipelineError(f"Step 0 LLM response missing '{key}'.")
        result.setdefault("education", [])
        result.setdefault("projects", [])
        result.setdefault("certifications", [])
        result.setdefault("global_missing_information", [])
        result.setdefault("total_years_experience", None)
        result.setdefault("seniority_signal", "unknown")
        result.setdefault("resume_language", "en")
        result.setdefault("primary_domain", None)
        return result

    # ---- Step 1: JD requirements (LLM) ----

    def step1_jd_requirements(self, jd_text: str) -> dict[str, Any]:
        """Extract required skills and role metadata from a JD."""
        self._validate_text(jd_text, "Job Description")
        prompt = AgentPrompts.jd_requirements(jd_text)
        result = self._call_llm_for_json(prompt, step_name="Step 1 (JD Requirements)")

        if "required_skills" not in result:
            raise PipelineError("Step 1 LLM response missing 'required_skills'.")
        result.setdefault("global_missing_information", [])
        result.setdefault("role_title", None)
        result.setdefault("role_family", "other")
        result.setdefault("role_seniority", "unknown")
        result.setdefault("responsibilities", [])
        for skill in result.get("required_skills", []):
            if isinstance(skill, dict):
                skill.setdefault("skill_kind", "hard")
                skill.setdefault("category", "other")
        return result

    # ---- Step 1B: Semantic skill match (LLM) ----

    def step1b_semantic_skill_match(
        self,
        jd_requirements: dict[str, Any],
        parsed_resume: dict[str, Any],
    ) -> dict[str, Any]:
        """Map each JD-required skill to the best-matching resume evidence."""
        prompt = AgentPrompts.semantic_skill_match(
            required_skills_json=json.dumps(jd_requirements["required_skills"], indent=2),
            parsed_resume_json=json.dumps(parsed_resume, indent=2),
        )
        result = self._call_llm_for_json(prompt, step_name="Step 1B (Semantic Skill Match)")

        if "skill_matches" not in result:
            raise PipelineError("Step 1B LLM response missing 'skill_matches'.")
        result.setdefault("global_missing_information", [])
        return result

    # ---- Step 2: Match percentage (code) ----

    def step2_match_percentage(
        self,
        jd_requirements: dict[str, Any],
        skill_matches: dict[str, Any],
    ) -> dict[str, Any]:
        """Thin wrapper around ``compute_match_percentage``."""
        return compute_match_percentage(jd_requirements, skill_matches)

    # ---- Step 3: Gap classification (code) ----

    def step3_gap_classification(self, match_percentage: float) -> str:
        """Return the Low/Medium/High gap band for ``match_percentage``."""
        try:
            return gap_classification(match_percentage)
        except ValueError as e:
            raise PipelineError(str(e)) from e

    # ---- Step 4A: Generate assessment question (LLM) ----

    def step4_generate_question(
        self,
        skill_name: str,
        jd_requirement: dict[str, Any],
        resume_match: dict[str, Any],
        previous_qa: list[tuple[str, str]] | None = None,
        role_seniority: str = "unknown",
        skill_kind: str = "hard",
    ) -> dict[str, Any]:
        """Ask the LLM for the next assessment question for ``skill_name``."""
        prev = self._format_previous_qa(previous_qa or [])
        prompt = AgentPrompts.assessment_question(
            skill_name=skill_name,
            jd_requirement_json=json.dumps(jd_requirement, indent=2),
            resume_match_json=json.dumps(resume_match, indent=2),
            previous_qa=prev,
            role_seniority=role_seniority,
            skill_kind=skill_kind,
        )
        result = self._call_llm_for_json(prompt, step_name="Step 4A (Assessment Question)")

        if "question" not in result:
            raise PipelineError("Step 4A LLM response missing 'question'.")
        result.setdefault("question_type", "general")
        result.setdefault("difficulty", "intermediate")
        result.setdefault("target_signals", [])
        result.setdefault("why_this_question", "")
        return result

    # ---- Step 4B: Evaluate response (LLM + code) ----

    def step4_evaluate_response(
        self,
        skill_name: str,
        question: str,
        response_text: str,
        expected_level: int = 3,
        skill_kind: str = "hard",
    ) -> dict[str, Any]:
        """Score a candidate response across calibrated dimensions."""
        prompt = AgentPrompts.response_evaluation(
            skill_name=skill_name,
            question=question,
            response=response_text,
            expected_level=expected_level,
            skill_kind=skill_kind,
        )
        result = self._call_llm_for_json(prompt, step_name="Step 4B (Response Evaluation)")

        dims = result.get("dimension_scores", {}) or {}
        if skill_kind == "soft":
            expected_keys = ["concreteness", "outcome", "self_awareness"]
        else:
            expected_keys = [
                "relevance",
                "specificity",
                "technical_correctness",
                "depth",
                "experience_signal",
            ]
        # Clamp each dimension score into the rubric range 1-5.
        for key in expected_keys:
            dims[key] = max(1, min(5, int(dims.get(key, 1))))

        if skill_kind == "soft":
            overall_score = round(
                (dims["concreteness"] + dims["outcome"] + dims["self_awareness"]) / 3.0,
                2,
            )
        else:
            overall_score = round(
                0.20 * dims["relevance"]
                + 0.20 * dims["specificity"]
                + 0.25 * dims["technical_correctness"]
                + 0.20 * dims["depth"]
                + 0.15 * dims["experience_signal"],
                2,
            )

        result["dimension_scores"] = dims
        result["overall_score"] = overall_score
        result["provisional_level"] = max(1, min(5, int(result.get("provisional_level", 1))))
        result.setdefault("confidence", "low")
        result.setdefault("strengths", [])
        result.setdefault("weaknesses", [])
        result.setdefault("should_ask_follow_up", False)
        result.setdefault("follow_up_focus", "")
        result.setdefault("reasoning", "")
        result.setdefault("is_relevant", True)
        return result

    # ---- Step 5: Revised gap analysis (code) ----

    def step5_revised_gap_analysis(
        self,
        jd_requirements: dict[str, Any],
        skill_matches: dict[str, Any],
        assessment_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Thin wrapper around ``build_validated_gaps``."""
        return build_validated_gaps(jd_requirements, skill_matches, assessment_results)

    # ---- Step 6: Learning plan (LLM) ----

    def step6_learning_plan(
        self,
        validated_gaps: dict[str, Any],
        role_family: str = "other",
        primary_domain: str | None = None,
        time_budget_hours_per_week: int = 5,
    ) -> dict[str, Any]:
        """Produce a role-aware, budget-bounded learning plan."""
        prompt = AgentPrompts.learning_plan(
            json.dumps(validated_gaps, indent=2),
            role_family=role_family,
            primary_domain=primary_domain,
            time_budget_hours_per_week=time_budget_hours_per_week,
        )
        result = self._call_llm_for_json(prompt, step_name="Step 6 (Learning Plan)")

        if "plan_items" not in result:
            raise PipelineError("Step 6 LLM response missing 'plan_items'.")

        result.setdefault("adjacent_skills", [])

        # Belt-and-suspenders: drop plan_items for skills already marked as met.
        met_skills = {
            str(s.get("skill_name", "")).strip().lower()
            for s in validated_gaps.get("revised_skills", [])
            if isinstance(s, dict) and s.get("gap_status") == "met"
        }
        if met_skills:
            result["plan_items"] = [
                item
                for item in result.get("plan_items", [])
                if isinstance(item, dict)
                and str(item.get("skill_name", "")).strip().lower() not in met_skills
            ]

        # Normalise plan items: default priority, clamp hours, drop invalid URLs.
        for item in result.get("plan_items", []):
            if not isinstance(item, dict):
                continue
            item.setdefault("priority", "medium")
            item["estimated_hours"] = _clamp_hours(item.get("estimated_hours"))
            for resource in item.get("resources", []) or []:
                if not isinstance(resource, dict):
                    continue
                resource.setdefault("cost_tier", "freemium")
                url = resource.get("url")
                if url and not _is_allowed_url(str(url)):
                    resource.pop("url", None)

            # If no resource has a URL left, append a canonical fallback.
            resources = item.get("resources", []) or []
            has_any_url = any(
                isinstance(r, dict) and r.get("url") for r in resources
            )
            if not has_any_url:
                fallback = _fallback_url_for_skill(str(item.get("skill_name", "")))
                if fallback:
                    title, url = fallback
                    resources.append({
                        "title": title,
                        "type": "docs",
                        "level": "beginner",
                        "cost_tier": "free",
                        "url": url,
                        "reason": "Canonical starting point for this skill.",
                    })
                    item["resources"] = resources

        # Same normalisation for adjacent skills.
        for item in result.get("adjacent_skills", []):
            if not isinstance(item, dict):
                continue
            if "estimated_hours" in item:
                item["estimated_hours"] = _clamp_hours(item.get("estimated_hours"))
            for resource in item.get("resources", []) or []:
                if not isinstance(resource, dict):
                    continue
                resource.setdefault("cost_tier", "freemium")
                url = resource.get("url")
                if url and not _is_allowed_url(str(url)):
                    resource.pop("url", None)

        # Cap total plan hours against a 16-week horizon at the learner's budget.
        max_total = max(float(time_budget_hours_per_week) * 16.0, 16.0)
        total = sum(float(item.get("estimated_hours", 0)) for item in result.get("plan_items", []))
        total += sum(float(item.get("estimated_hours", 0)) for item in result.get("adjacent_skills", []))
        if total > max_total:
            # Over budget: scale every item's hours proportionally down to the cap.
            scale = max_total / total
            for item in result.get("plan_items", []):
                if isinstance(item, dict):
                    item["estimated_hours"] = _clamp_hours(float(item.get("estimated_hours", 0)) * scale)
            for item in result.get("adjacent_skills", []):
                if isinstance(item, dict) and "estimated_hours" in item:
                    item["estimated_hours"] = _clamp_hours(float(item.get("estimated_hours", 0)) * scale)
            total = sum(float(item.get("estimated_hours", 0)) for item in result.get("plan_items", []))
            total += sum(float(item.get("estimated_hours", 0)) for item in result.get("adjacent_skills", []))
        result["total_estimated_hours"] = round(total, 1)
        return result

    # ---- Helpers ----

    @staticmethod
    def _validate_text(text: str, field_name: str) -> None:
        """Raise ``PipelineError`` when ``text`` is empty or whitespace-only."""
        if text is None or not text.strip():
            raise PipelineError(f"{field_name} is empty. Please provide a non-empty {field_name}.")

    @staticmethod
    def _format_previous_qa(previous_qa: list[tuple[str, str]]) -> str:
        """Format previous Q/A turns for prompt injection."""
        if not previous_qa:
            return "(none)"
        lines = []
        for i, (q, a) in enumerate(previous_qa, 1):
            lines.append(f"Q{i}: {q}\nA{i}: {a}")
        return "\n\n".join(lines)

    def _call_llm_for_json(
        self,
        prompt: str,
        step_name: str,
        max_retries: int = 1,
    ) -> dict[str, Any]:
        """Call the LLM for JSON, retrying once with a stricter reminder."""
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.llm.generate_json(prompt, temperature=0.2)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < max_retries:
                    prompt = (
                        prompt
                        + "\n\nIMPORTANT: Return ONLY valid JSON. "
                        "No markdown, no commentary, no preamble."
                    )
        raise PipelineError(
            f"{step_name} failed: could not parse LLM JSON response. Last error: {last_error}"
        )

    # ---- Bonus: Generate JD (LLM) ----

    def generate_jd(self, company: str, role: str, seniority: str = "") -> str:
        """Generate a realistic JD for the given company and role."""
        if not company.strip() or not role.strip():
            raise PipelineError("Company and role are both required.")

        prompt = AgentPrompts.generate_jd(
            company=company.strip(),
            role=role.strip(),
            seniority=seniority.strip(),
        )
        response = self.llm.generate(prompt, temperature=0.3)
        text = response.text.strip()

        if not text:
            raise PipelineError("LLM returned empty JD.")

        return self._strip_grounding_sections(text)

    @staticmethod
    def _strip_grounding_sections(text: str) -> str:
        """Strip meta sections ("Grounding Mode", "Source Notes") from the JD.

        Removes the heading and every following line until the next known
        section heading (Job Title / Company Overview / ...).
        """
        section_headings = {
            "job title",
            "company overview",
            "role summary",
            "key responsibilities",
            "required qualifications",
            "preferred qualifications",
        }
        skip_headings = {"grounding mode", "source notes"}

        output_lines: list[str] = []
        skipping = False
        for line in text.splitlines():
            stripped = line.strip().lower().rstrip(":")
            if stripped in skip_headings:
                skipping = True
                continue
            if skipping:
                # Stop skipping once the next known section heading appears.
                if stripped in section_headings:
                    skipping = False
                    output_lines.append(line)
                # Otherwise keep skipping.
                continue
            output_lines.append(line)

        # Drop any leading blank lines introduced by the strip.
        while output_lines and not output_lines[0].strip():
            output_lines.pop(0)

        return "\n".join(output_lines)
