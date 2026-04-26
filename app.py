"""Streamlit entry point for the Skill Assessment & Learning Plan Agent.

Single-page app with state-driven views:
    input -> skill_overview -> assessment -> learning_plan
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# Load .env for local development. In Streamlit Cloud, secrets are set via the UI.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from core.exporter import Exporter
from core.llm_provider import OpenRouterProvider
from core.pipeline import Pipeline, PipelineError
from core.resume_parser import (
    ResumeParser,
    UnreadableFileError,
    UnsupportedFormatError,
)

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Skill Assessment & Learning Plan Agent",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% 0%, rgba(34, 197, 94, 0.08), transparent 60%),
            radial-gradient(1000px 500px at 100% 100%, rgba(59, 130, 246, 0.06), transparent 60%),
            linear-gradient(180deg, #0A0F14 0%, #0D1520 60%, #0A1410 100%) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0C1218 0%, #0A0F14 100%) !important;
    }
    .stApp [data-testid="stHeader"] {
        background: transparent !important;
    }

    .stApp textarea,
    .stApp [data-testid="stTextArea"] textarea {
        background-color: #111821 !important;
        color: #E6EDF3 !important;
        border: 1px solid rgba(34, 197, 94, 0.35) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.08) inset,
                    0 2px 12px rgba(0, 0, 0, 0.25) !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    }
    .stApp textarea:focus,
    .stApp [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(34, 197, 94, 0.85) !important;
        box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.25),
                    0 2px 16px rgba(34, 197, 94, 0.15) !important;
        outline: none !important;
    }
    .stApp textarea::placeholder {
        color: rgba(230, 237, 243, 0.35) !important;
    }

    .stApp [data-testid="stTextInput"] input {
        background-color: #111821 !important;
        color: #E6EDF3 !important;
        border: 1px solid rgba(34, 197, 94, 0.30) !important;
        border-radius: 8px !important;
    }
    .stApp [data-testid="stTextInput"] input:focus {
        border-color: rgba(34, 197, 94, 0.85) !important;
        box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.20) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Session state initialisation
# =============================================================================


def _init_state() -> None:
    defaults = {
        "step": "input",
        "jd_text": "",
        "resume_text": "",
        "resume_filename": "",
        # Pipeline outputs (v2).
        "parsed_resume": None,
        "jd_requirements": None,
        "skill_matches": None,
        "match_results": None,
        "gap_band": None,
        "validated_gaps": None,
        "learning_plan": None,
        # Assessment UI state.
        "assessment_state": {
            "current_skill_index": 0,
            "current_question_number": 0,
            "current_question_payload": None,
            "per_skill_qa": [],
            "results": [],
            "completed": False,
        },
        "api_key": "",
        "model_name": "openai/gpt-oss-120b:free",
        "error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

# =============================================================================
# Helpers
# =============================================================================


def _load_api_key() -> str:
    """Resolve the OpenRouter API key from Streamlit secrets or env.

    The key is NOT surfaced in the UI — it lives in `.env` locally or
    Streamlit Cloud secrets for deploys.
    """
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")


OPENROUTER_MODELS = [
    "openai/gpt-oss-120b:free",
    "minimax/minimax-m2.5:free",
    "z-ai/glm-4.5-air:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]


def _get_pipeline() -> Pipeline | None:
    key = _load_api_key()
    if not key:
        st.error(
            "OpenRouter API key missing. Set `OPENROUTER_API_KEY` in your local "
            "`.env` file or in Streamlit Cloud secrets."
        )
        return None

    model = st.session_state.get("model_name") or OPENROUTER_MODELS[0]
    try:
        llm = OpenRouterProvider(api_key=key, model=model)
    except Exception as e:
        st.error(f"Could not initialise OpenRouter: {e}")
        return None
    return Pipeline(llm=llm)


def _goto(step: str) -> None:
    st.session_state["step"] = step
    st.session_state["error"] = ""


def _all_jd_skills() -> list[dict]:
    jd = st.session_state.get("jd_requirements") or {}
    return jd.get("required_skills", [])


def _resume_match_for_skill(skill_name: str) -> dict:
    matches = st.session_state.get("skill_matches", {}).get("skill_matches", [])
    for item in matches:
        if item.get("skill_name", "").strip().lower() == skill_name.strip().lower():
            return item
    return {
        "skill_name": skill_name,
        "matched": False,
        "match_type": "no_match",
        "matched_resume_terms": [],
        "evidence_strength": "none",
        "years_experience": None,
        "evidence_spans": [],
        "reasoning": "",
        "confidence": "low",
    }


def _reset_assessment_state() -> None:
    st.session_state["assessment_state"] = {
        "current_skill_index": 0,
        "current_question_number": 0,
        "current_question_payload": None,
        "per_skill_qa": [],
        "results": [],
        "completed": False,
    }



# =============================================================================
# Sidebar
# =============================================================================


def _sidebar() -> None:
    st.sidebar.title("🎯 Skill Agent")
    st.sidebar.caption("AI-powered skill assessment + learning plan")

    with st.sidebar.expander("⚙️ LLM Setup", expanded=False):
        if st.session_state.get("model_name") not in OPENROUTER_MODELS:
            st.session_state["model_name"] = OPENROUTER_MODELS[0]
        st.selectbox(
            "Choose a model",
            options=OPENROUTER_MODELS,
            key="model_name",
            help="Pick one of the available free-tier models to power the assessment.",
        )
        if not _load_api_key():
            st.caption("⚠️ Set `OPENROUTER_API_KEY` in your `.env` or Streamlit secrets.")

    sample_dir = Path(__file__).parent / "data" / "samples"
    if sample_dir.exists():
        samples = sorted(sample_dir.glob("*_jd.txt"))
        sample_names = [s.stem.replace("_jd", "") for s in samples]
        if sample_names:
            with st.sidebar.expander("📄 Try a sample", expanded=False):
                chosen = st.selectbox(
                    "Pick a sample",
                    options=sample_names,
                    key="sample_choice",
                    help="Pre-loaded JD + resume pairs. Useful for a quick demo.",
                )
                jd_path = sample_dir / f"{chosen}_jd.txt"
                resume_path = sample_dir / f"{chosen}_resume.txt"
                if st.button("Load this sample", use_container_width=True):
                    st.session_state["jd_text"] = jd_path.read_text() if jd_path.exists() else ""
                    st.session_state["resume_text"] = (
                        resume_path.read_text() if resume_path.exists() else ""
                    )
                    st.session_state["resume_filename"] = f"{chosen}_resume.txt"
                    _goto("input")
                    st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("**Progress**")
    steps = [
        ("input", "1. Input"),
        ("skill_overview", "2. Skill Analysis"),
        ("assessment", "3. Assessment"),
        ("learning_plan", "4. Learning Plan"),
    ]
    current = st.session_state["step"]
    for key, label in steps:
        prefix = (
            "✅"
            if steps.index((key, label)) < [s[0] for s in steps].index(current)
            else ("▶️" if key == current else "◻️")
        )
        st.sidebar.markdown(f"{prefix} {label}")

    st.sidebar.divider()
    if st.sidebar.button("🔄 Start over", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k not in ("model_name",):
                del st.session_state[k]
        _init_state()
        st.rerun()


# =============================================================================
# Views
# =============================================================================


def _view_input() -> None:
    st.title("🎯 Skill Assessment & Learning Plan Agent")
    st.caption(
        "Upload your resume, paste a job description, and get a conversational "
        "skill assessment plus a personalised learning plan."
    )

    col_jd, col_resume = st.columns(2)

    with col_jd:
        st.subheader("Job Description")
        jd_tabs = st.tabs(["📝 Paste", "⬆️ Upload", "🤖 Fetch by role"])

        with jd_tabs[0]:
            st.session_state["jd_text"] = st.text_area(
                "Paste the JD here",
                value=st.session_state.get("jd_text", ""),
                height=300,
                placeholder="e.g. Senior Backend Engineer at FinCo...",
                label_visibility="collapsed",
            )

        with jd_tabs[1]:
            jd_file = st.file_uploader(
                "Upload JD (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                key="jd_file_uploader",
            )
            if jd_file is not None:
                try:
                    parser = ResumeParser()
                    st.session_state["jd_text"] = parser.parse(
                        jd_file.getvalue(), jd_file.name
                    )
                    st.toast(f"✅ Parsed {jd_file.name}", icon="📄")
                except (UnsupportedFormatError, UnreadableFileError) as e:
                    st.error(str(e))

        with jd_tabs[2]:
            st.caption("Let the AI draft a realistic JD for any company + role.")
            fetch_company = st.text_input(
                "Company", placeholder="e.g. Stripe, Netflix, Amazon", key="fetch_company"
            )
            fetch_role = st.text_input(
                "Role", placeholder="e.g. Senior Data Scientist", key="fetch_role"
            )
            fetch_seniority = st.selectbox(
                "Seniority (optional)",
                options=["", "Junior", "Mid", "Senior", "Staff", "Principal"],
                key="fetch_seniority",
            )
            if st.button("✨ Generate JD", use_container_width=True):
                if not fetch_company.strip() or not fetch_role.strip():
                    st.error("Enter both company and role.")
                else:
                    pipeline = _get_pipeline()
                    if pipeline:
                        try:
                            with st.spinner("Drafting JD..."):
                                st.session_state["jd_text"] = pipeline.generate_jd(
                                    fetch_company, fetch_role, fetch_seniority
                                )
                            st.toast(
                                f"✅ JD generated for {fetch_role} at {fetch_company}",
                                icon="✨",
                            )
                            st.rerun()
                        except PipelineError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Could not generate JD: {e}")

    with col_resume:
        st.subheader("Your Resume")
        resume_tabs = st.tabs(["📝 Paste", "⬆️ Upload"])

        with resume_tabs[0]:
            st.session_state["resume_text"] = st.text_area(
                "Paste your resume here",
                value=st.session_state.get("resume_text", ""),
                height=300,
                placeholder="Paste the full text of your resume...",
                label_visibility="collapsed",
            )

        with resume_tabs[1]:
            uploaded = st.file_uploader(
                "Upload resume (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                key="resume_file_uploader",
            )
            if uploaded is not None:
                try:
                    parser = ResumeParser()
                    st.session_state["resume_text"] = parser.parse(
                        uploaded.getvalue(), uploaded.name
                    )
                    st.session_state["resume_filename"] = uploaded.name
                    st.toast(f"✅ Parsed {uploaded.name}", icon="📄")
                except (UnsupportedFormatError, UnreadableFileError) as e:
                    st.error(str(e))

    st.divider()

    if st.button("Analyze →", type="primary", use_container_width=True):
        if not st.session_state["jd_text"].strip():
            st.error("Please provide a job description.")
            return
        if not st.session_state["resume_text"].strip():
            st.error("Please provide a resume (upload or paste).")
            return

        pipeline = _get_pipeline()
        if not pipeline:
            return

        try:
            with st.spinner("Parsing resume..."):
                st.session_state["parsed_resume"] = pipeline.step0_parse_resume(
                    st.session_state["resume_text"]
                )

            with st.spinner("Extracting JD requirements..."):
                st.session_state["jd_requirements"] = pipeline.step1_jd_requirements(
                    st.session_state["jd_text"]
                )

            with st.spinner("Matching resume to JD skills..."):
                st.session_state["skill_matches"] = pipeline.step1b_semantic_skill_match(
                    st.session_state["jd_requirements"],
                    st.session_state["parsed_resume"],
                )

            with st.spinner("Calculating overall match..."):
                st.session_state["match_results"] = pipeline.step2_match_percentage(
                    st.session_state["jd_requirements"],
                    st.session_state["skill_matches"],
                )

            pct = st.session_state["match_results"]["overall_match_percentage"]
            st.session_state["gap_band"] = pipeline.step3_gap_classification(pct)

            _reset_assessment_state()
            st.session_state["validated_gaps"] = None
            st.session_state["learning_plan"] = None

            _goto("skill_overview")
            st.rerun()

        except PipelineError as e:
            st.error(f"Pipeline error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


def _view_skill_overview() -> None:
    st.title("Skill Analysis Results")

    match_results = st.session_state.get("match_results") or {}
    jd_requirements = st.session_state.get("jd_requirements") or {}
    skill_matches = st.session_state.get("skill_matches") or {}

    pct = match_results.get("overall_match_percentage", 0.0)
    gap_band = st.session_state.get("gap_band", "Unknown")

    matches = skill_matches.get("skill_matches", [])
    required_skills = jd_requirements.get("required_skills", [])
    matched_count = sum(1 for m in matches if m.get("matched"))
    unmatched_count = max(0, len(required_skills) - matched_count)

    gap_icons = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}

    col1, col2, col3 = st.columns(3)
    col1.metric("Resume Match", f"{pct:.1f}%")
    col2.metric("Gap Classification", f"{gap_icons.get(gap_band, '')} {gap_band}")
    col3.metric("Skills", f"{matched_count} matched / {unmatched_count} not matched")

    st.divider()

    import pandas as pd

    if required_skills:
        st.subheader("📌 JD Requirements")
        df_req = pd.DataFrame(required_skills)
        # Drop ghost rows (all-NaN or missing skill_name) before render.
        if "skill_name" in df_req.columns:
            df_req = df_req.dropna(subset=["skill_name"]).reset_index(drop=True)
        req_cols = [
            c
            for c in [
                "skill_name",
                "category",
                "requirement_type",
                "required_level",
                "years_required",
                "importance",
                "jd_evidence_span",
            ]
            if c in df_req.columns
        ]
        st.dataframe(
            df_req[req_cols],
            hide_index=True,
            height=min(400, 38 * (len(df_req) + 1) + 3),
            column_config={
                "skill_name": st.column_config.TextColumn(
                    "skill_name", width=220
                ),
                "jd_evidence_span": st.column_config.TextColumn(
                    "jd_evidence_span", width=700
                ),
            },
        )

    if matches:
        st.subheader("🔎 Resume Skill Matches")
        df_match = pd.DataFrame(matches)
        # Drop ghost rows (all-NaN or missing skill_name) before render.
        if "skill_name" in df_match.columns:
            df_match = df_match.dropna(subset=["skill_name"]).reset_index(drop=True)
        match_cols = [
            c
            for c in [
                "skill_name",
                "matched",
                "evidence_strength",
                "years_experience",
                "matched_resume_terms",
                "reasoning",
            ]
            if c in df_match.columns
        ]
        if "matched_resume_terms" in df_match.columns:
            df_match["matched_resume_terms"] = df_match["matched_resume_terms"].apply(
                lambda v: ", ".join(v) if isinstance(v, list) else v
            )
        st.dataframe(
            df_match[match_cols],
            hide_index=True,
            height=min(400, 38 * (len(df_match) + 1) + 3),
            column_config={
                "skill_name": st.column_config.TextColumn(
                    "skill_name", width=220
                ),
                "matched_resume_terms": st.column_config.TextColumn(
                    "matched_resume_terms", width=280
                ),
                "reasoning": st.column_config.TextColumn(
                    "reasoning", width=700
                ),
            },
        )

    breakdown = match_results.get("per_skill_breakdown", [])
    if breakdown:
        st.subheader("📊 Weighted Match Breakdown")
        df_break = pd.DataFrame(breakdown)
        # Drop ghost rows (all-NaN or missing skill_name) before render.
        if "skill_name" in df_break.columns:
            df_break = df_break.dropna(subset=["skill_name"]).reset_index(drop=True)
        break_cols = [
            c
            for c in [
                "skill_name",
                "requirement_type",
                "importance",
                "required_level",
                "evidence_strength",
                "score",
                "weight",
                "weighted_score",
            ]
            if c in df_break.columns
        ]
        st.dataframe(
            df_break[break_cols],
            hide_index=True,
            height=min(400, 38 * (len(df_break) + 1) + 3),
            column_config={
                "skill_name": st.column_config.TextColumn(
                    "skill_name", width=220
                ),
            },
        )

    st.divider()

    col_back, col_next = st.columns(2)
    if col_back.button("← Back to input"):
        _goto("input")
        st.rerun()

    if col_next.button("Start Assessment →", type="primary"):
        _reset_assessment_state()
        _goto("assessment")
        st.rerun()



def _view_assessment() -> None:
    st.title("💬 Conversational Assessment")

    skills = _all_jd_skills()
    if not skills:
        st.warning("No JD skills available to assess. Go back to input.")
        if st.button("← Back"):
            _goto("skill_overview")
            st.rerun()
        return

    state = st.session_state["assessment_state"]
    idx = state["current_skill_index"]

    st.progress(
        min(idx / len(skills), 1.0) if skills else 0.0,
        text=f"Skill {min(idx + 1, len(skills))} of {len(skills)}",
    )

    if idx >= len(skills):
        state["completed"] = True
        pipeline = _get_pipeline()
        if not pipeline:
            return
        try:
            with st.spinner("Validating gaps from assessment..."):
                st.session_state["validated_gaps"] = pipeline.step5_revised_gap_analysis(
                    jd_requirements=st.session_state["jd_requirements"],
                    skill_matches=st.session_state["skill_matches"],
                    assessment_results=state["results"],
                )
            with st.spinner("Generating learning plan..."):
                st.session_state["learning_plan"] = pipeline.step6_learning_plan(
                    st.session_state["validated_gaps"]
                )
            _goto("learning_plan")
            st.rerun()
        except PipelineError as e:
            st.error(f"Pipeline error: {e}")
            if st.button("Retry"):
                st.rerun()
        return

    current_skill = skills[idx]
    skill_name = current_skill["skill_name"]
    required = current_skill.get("required_level", 3)
    resume_match = _resume_match_for_skill(skill_name)

    st.subheader(f"Skill: {skill_name}")
    st.caption(
        f"Required level: {required}/5 · "
        f"Resume match: {resume_match.get('match_type', 'no_match')} · "
        f"Evidence: {resume_match.get('evidence_strength', 'none')}"
    )

    pipeline = _get_pipeline()
    if not pipeline:
        return

    if not state["current_question_payload"]:
        state["current_question_number"] += 1
        try:
            with st.spinner("Thinking of a good question..."):
                previous = [(qa["question"], qa["response"]) for qa in state["per_skill_qa"]]
                state["current_question_payload"] = pipeline.step4_generate_question(
                    skill_name=skill_name,
                    jd_requirement=current_skill,
                    resume_match=resume_match,
                    previous_qa=previous,
                )
        except Exception as e:
            st.error(f"Could not generate question: {e}")
            return

    question_payload = state["current_question_payload"]
    question_text = question_payload["question"]

    st.caption(
        f"Question type: {question_payload.get('question_type', 'general')} · "
        f"Difficulty: {question_payload.get('difficulty', 'intermediate')}"
    )

    for qa in state["per_skill_qa"]:
        with st.chat_message("assistant"):
            st.write(qa["question"])
        with st.chat_message("user"):
            st.write(qa["response"])
        with st.expander("Evaluation details", expanded=False):
            st.write(f"Overall score: {qa.get('overall_score', '')}")
            st.write(f"Provisional level: {qa.get('provisional_level', '')}")
            st.write(f"Confidence: {qa.get('confidence', '')}")
            dims = qa.get("dimension_scores", {})
            if dims:
                st.json(dims)
            strengths = qa.get("strengths", [])
            weaknesses = qa.get("weaknesses", [])
            if strengths:
                st.write("Strengths:")
                for s in strengths:
                    st.markdown(f"- {s}")
            if weaknesses:
                st.write("Weaknesses:")
                for w in weaknesses:
                    st.markdown(f"- {w}")

    with st.chat_message("assistant"):
        st.write(question_text)

    response = st.chat_input("Your answer (or type 'skip' to skip this skill)")
    col_skip, _ = st.columns([1, 4])
    if col_skip.button("Skip this skill"):
        response = "skip"

    if response:
        response_lower = response.strip().lower()

        if response_lower == "skip":
            state["results"].append(
                {
                    "skill_name": skill_name,
                    "overall_score": 1.0,
                    "provisional_level": 1,
                    "confidence": "low",
                    "dimension_scores": {
                        "relevance": 1,
                        "specificity": 1,
                        "technical_correctness": 1,
                        "depth": 1,
                        "experience_signal": 1,
                    },
                    "is_relevant": False,
                    "strengths": [],
                    "weaknesses": ["Candidate skipped this skill."],
                    "should_ask_follow_up": False,
                    "follow_up_focus": "",
                    "reasoning": "Candidate chose to skip this skill.",
                }
            )
            state["current_skill_index"] += 1
            state["current_question_number"] = 0
            state["current_question_payload"] = None
            state["per_skill_qa"] = []
            st.rerun()
            return

        try:
            with st.spinner("Evaluating..."):
                evaluation = pipeline.step4_evaluate_response(
                    skill_name=skill_name,
                    question=question_text,
                    response_text=response,
                )
        except Exception as e:
            st.error(f"Could not evaluate response: {e}")
            return

        state["per_skill_qa"].append(
            {
                "question": question_text,
                "response": response,
                "overall_score": evaluation["overall_score"],
                "provisional_level": evaluation["provisional_level"],
                "dimension_scores": evaluation["dimension_scores"],
                "is_relevant": evaluation.get("is_relevant", True),
                "confidence": evaluation.get("confidence", "low"),
                "strengths": evaluation.get("strengths", []),
                "weaknesses": evaluation.get("weaknesses", []),
            }
        )

        state["results"].append({"skill_name": skill_name, **evaluation})

        q_count = state["current_question_number"]
        should_follow_up = evaluation.get("should_ask_follow_up", False)

        if q_count >= 3 or not should_follow_up:
            state["current_skill_index"] += 1
            state["current_question_number"] = 0
            state["current_question_payload"] = None
            state["per_skill_qa"] = []
        else:
            state["current_question_payload"] = None

        st.rerun()


def _view_learning_plan() -> None:
    st.title("📚 Your Personalised Learning Plan")

    validated = st.session_state.get("validated_gaps") or {}
    plan = st.session_state.get("learning_plan") or {}

    initial_pct = st.session_state.get("match_results", {}).get("overall_match_percentage", 0)
    final_pct = (validated or {}).get("final_match_percentage")
    skills_needing_attention = len(validated.get("skills_needing_attention", []))
    total_hours = plan.get("total_estimated_hours", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resume Match", f"{initial_pct:.1f}%")
    col2.metric(
        "Final Match",
        f"{final_pct:.1f}%" if isinstance(final_pct, (int, float)) else "—",
    )
    col3.metric("Skills Needing Attention", skills_needing_attention)
    col4.metric("Total Learning Time", f"{total_hours:.0f} hours")

    st.divider()

    revised_skills = validated.get("revised_skills", [])
    if revised_skills:
        st.subheader("🎯 Validated Gap Analysis")
        import pandas as pd

        df = pd.DataFrame(revised_skills)
        # Drop ghost rows (all-NaN or missing skill_name) before render.
        if "skill_name" in df.columns:
            df = df.dropna(subset=["skill_name"]).reset_index(drop=True)
        display_cols = [
            c
            for c in [
                "skill_name",
                "importance",
                "required_level",
                "validated_current_level",
                "gap_status",
                "resume_evidence_strength",
                "assessment_turns_count",
            ]
            if c in df.columns
        ]
        st.dataframe(
            df[display_cols],
            hide_index=True,
            height=min(400, 38 * (len(df) + 1) + 3),
            column_config={
                "skill_name": st.column_config.TextColumn(
                    "skill_name", width=220
                ),
            },
        )

    st.divider()

    plan_items = plan.get("plan_items", [])
    if plan_items:
        st.subheader("📖 Learning Plan")
        for item in plan_items:
            gap_type = item.get("gap_type", "").replace("_", " ").title()
            hours = item.get("estimated_hours", 0)
            with st.expander(
                f"**{item.get('skill_name', '')}** — {gap_type} · ~{hours}h",
                expanded=False,
            ):
                st.write(
                    f"**Current:** {item.get('current_level', '')}/5 → "
                    f"**Target:** {item.get('target_level', '')}/5"
                )
                st.write(f"**Priority:** {item.get('priority', '').title()}")
                st.write(f"**Why this matters:** {item.get('why_this_matters', '')}")
                st.write(f"**Objective:** {item.get('learning_objective', '')}")

                deps = item.get("dependency_skills", [])
                if deps:
                    st.write(f"**Dependencies:** {', '.join(deps)}")

                st.write(f"**Practice task:** {item.get('practice_task', '')}")
                st.write(f"**Checkpoint:** {item.get('checkpoint', '')}")

                st.write("**Resources:**")
                for r in item.get("resources", []):
                    title = r.get("title", "")
                    rtype = r.get("type", "")
                    level = r.get("level", "")
                    reason = r.get("reason", "")
                    url = r.get("url", "")
                    if url:
                        st.markdown(f"- [{title}]({url}) — _{rtype}, {level}_")
                    else:
                        st.markdown(f"- **{title}** — _{rtype}, {level}_")
                    if reason:
                        st.caption(reason)

    adjacent = plan.get("adjacent_skills", [])
    if adjacent:
        st.subheader("✨ Adjacent Skills Worth Exploring")
        for item in adjacent:
            hours = item.get("estimated_hours", 0)
            with st.expander(f"**{item.get('skill_name', '')}** · ~{hours}h"):
                if item.get("why_this_matters"):
                    st.write(item["why_this_matters"])
                if item.get("learning_objective"):
                    st.write(f"**Objective:** {item['learning_objective']}")
                if item.get("practice_task"):
                    st.write(f"**Practice task:** {item['practice_task']}")
                if item.get("checkpoint"):
                    st.write(f"**Checkpoint:** {item['checkpoint']}")
                for r in item.get("resources", []):
                    title = r.get("title", "")
                    rtype = r.get("type", "")
                    level = r.get("level", "")
                    url = r.get("url", "")
                    if url:
                        st.markdown(f"- [{title}]({url}) — _{rtype}, {level}_")
                    else:
                        st.markdown(f"- **{title}** — _{rtype}, {level}_")

    st.divider()

    st.subheader("📥 Export Your Report")
    exporter = Exporter()
    session_data = {
        "parsed_resume": st.session_state.get("parsed_resume"),
        "jd_requirements": st.session_state.get("jd_requirements"),
        "skill_matches": st.session_state.get("skill_matches"),
        "match_results": st.session_state.get("match_results"),
        "gap_band": st.session_state.get("gap_band"),
        "validated_gaps": validated,
        "learning_plan": plan,
    }

    col_md, col_pdf = st.columns(2)

    try:
        md_text = exporter.to_markdown(session_data)
        col_md.download_button(
            "⬇️ Download Markdown",
            data=md_text,
            file_name="skill_assessment_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    except Exception as e:
        col_md.error(f"Markdown export unavailable: {e}")

    try:
        pdf_bytes = exporter.to_pdf(session_data)
        col_pdf.download_button(
            "⬇️ Download PDF",
            data=pdf_bytes,
            file_name="skill_assessment_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        col_pdf.error(f"PDF export unavailable: {e}")


# =============================================================================
# Main dispatch
# =============================================================================

_sidebar()

step = st.session_state["step"]
if step == "input":
    _view_input()
elif step == "skill_overview":
    _view_skill_overview()
elif step == "assessment":
    _view_assessment()
elif step == "learning_plan":
    _view_learning_plan()
else:
    st.error(f"Unknown step: {step}")
    _goto("input")
