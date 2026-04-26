# 🎯 Skill Assessment & Learning Plan AI Agent

An **AI agent** that reads a Job Description and a resume, then **conversationally assesses** the candidate on every required skill, validates real proficiency through multi-turn dialogue, and produces a **personalised learning plan** with curated resources, time estimates, and adjacent-skill suggestions.

Built for the _AI-Powered Skill Assessment & Personalised Learning Plan Agent_ hackathon.

## 🚀 Try It Live

**[Open the live demo →](https://your-app.streamlit.app)** _(replace with your Streamlit Cloud URL after deploy)_

Or run locally — see [Local Setup](#-local-setup) below.

## ✨ What the Agent Does

1. **Parses your resume** (PDF / DOCX / TXT) and a **pasted JD** — or generates a realistic JD for any company + role
2. **Extracts skills** from both sides with full metadata (category, required level, years, importance, role seniority, soft vs hard)
3. **Semantically matches** resume evidence to JD skills using recency decay, summary-only detection, and level-aware downgrading
4. Computes a **weighted resume match %** (importance × requirement type × evidence strength × years-gap modifier)
5. Runs a **conversational assessment** — 1–3 targeted questions per skill, calibrated to role seniority, with separate rubrics for soft skills
6. Emits a **validated gap analysis** with a final match % and per-skill breakdown of initial vs assessed proficiency
7. Generates a **personalised learning plan** — 2–5 resources per gap with working URLs, free/paid hints, time estimates, and adjacent-skill suggestions
8. **Exports** the full report as Markdown or PDF

## 🧠 The Agent Pipeline

The agent runs as a chain of LLM-driven steps + deterministic scoring. Every LLM step returns strict JSON validated by a schema, so the UI and downstream steps are decoupled from model variance.

| # | Step | Engine | What the agent does |
|---|------|--------|--------------------|
| 0 | Resume Parse | LLM | Extracts skills, seniority signal, total years, domain, language |
| 1 | JD Requirements | LLM | Extracts required skills with category enum, skill kind (hard/soft), role family, seniority |
| 1b | Semantic Skill Match | LLM | Matches resume evidence to JD skills with recency decay + level-aware rules |
| 2 | Weighted Match % | Code | Aggregates per-skill scores with importance + years-gap modifier |
| 3 | Gap Band | Code | Low / Medium / High classification |
| 4 | Conversational Assessment | LLM | Seniority-calibrated questions, fabrication penalty, soft-skill rubric |
| 5 | Validated Gap Analysis | LLM + Code | Revised proficiency levels + final match % computed from assessed levels |
| 6 | Learning Plan | LLM | Gap-ranked plan with curated resources, cost tier, time budget, canonical URL fallbacks |
| – | JD Generator | LLM | On-demand JD synthesis for any company + role + seniority |
| – | Export | Code | Markdown + PDF with safe long-token wrapping |

See [`core/agent_prompts.py`](core/agent_prompts.py) for the full prompt templates and [`core/pipeline.py`](core/pipeline.py) for the orchestration.

## 🏗 Architecture

```
Streamlit UI (app.py)
        │
        ▼
Pipeline Orchestrator (core/pipeline.py)
        │
        ├── Step 0:  Resume Parse        ┐
        ├── Step 1:  JD Requirements     │
        ├── Step 1b: Semantic Match      │  LLM-driven (strict JSON)
        ├── Step 4:  Assessment Q + Eval │
        ├── Step 5:  Revised Gaps        │
        ├── Step 6:  Learning Plan       ┘
        │
        ├── Step 2:  Weighted Scoring    ┐
        ├── Step 3:  Gap Classification  │  Deterministic (core/scoring.py)
        └── Final Match %                ┘
                       │
                       ▼
        LLMProvider — OpenRouter (default) · Gemini (fallback) · pluggable
```

- **Pluggable LLM** — swap providers by implementing `LLMProvider` in [`core/llm_provider.py`](core/llm_provider.py)
- **No database** — all session state lives in `st.session_state`
- **Deterministic scoring** — match % and gap bands are computed in code ([`core/scoring.py`](core/scoring.py)), so the LLM is only used where language understanding is needed

## 🎯 What Makes the Agent Reliable

- **Recency-aware evidence scoring** — a skill last used 5+ years ago is automatically downgraded
- **Fabrication penalty** — responses full of buzzwords with no how/why/when cap technical scores at 2
- **Seniority calibration** — expected proficiency scales with `role_seniority` (intern=1 → lead=5)
- **Soft-skill rubric** — for communication/leadership skills the agent swaps to `concreteness` / `outcome` / `self_awareness` dimensions instead of technical ones
- **Fallback resources** — if the LLM omits URLs, the pipeline appends a canonical starter (docs.python.org, react.dev, kubernetes.io, aws.amazon.com/documentation, etc.) based on skill keywords
- **Hours clamp** — every `plan_item` is capped at 3–30 hours and the total plan is proportionally scaled to fit 16 weeks of the user's time budget
- **Safe PDF export** — long URLs and skill names are soft-broken into 60-char chunks so `fpdf2.multi_cell` never crashes on unbreakable tokens

## 🛠 Local Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/SVMEKALINKESH/skill-assessment-agent.git
cd skill-assessment-agent

# 2. Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your LLM API key
cp .env.example .env
# Edit .env and paste ONE of:
#   OPENROUTER_API_KEY=...   (default, get at https://openrouter.ai/keys)
#   GEMINI_API_KEY=...       (fallback, get at https://aistudio.google.com/app/apikey)

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Supported LLM Models (OpenRouter free tier)

- `openai/gpt-oss-120b:free` (default)
- `minimax/minimax-m2.5:free`
- `z-ai/glm-4.5-air:free`
- `nvidia/nemotron-3-super-120b-a12b:free`

Pick one from the sidebar at runtime.

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to a **public GitHub repo**
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub
3. Click **New app**, select your repo, branch, and `app.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "your-key-here"
   ```
5. Click **Deploy** — done. Every `git push` auto-redeploys.

## 📄 Sample Inputs

The `data/samples/` folder ships with two ready-to-use JD + resume pairs:

- `swe_jd.txt` + `swe_resume.txt` — Senior Backend Engineer
- `ds_jd.txt` + `ds_resume.txt` — Data Scientist

Pick one from the sidebar to run the full pipeline without typing anything.

## 📁 Repo Structure

```
.
├── app.py                   # Streamlit UI — 4 state-driven views
├── core/
│   ├── agent_prompts.py     # All LLM prompt templates
│   ├── llm_provider.py      # LLMProvider ABC + OpenRouter + Gemini providers
│   ├── pipeline.py          # Agent orchestrator (steps 0-6 + JD gen)
│   ├── scoring.py           # Deterministic scoring + years-gap modifier + final match %
│   ├── resume_parser.py     # PDF / DOCX / TXT parsing
│   └── exporter.py          # Safe Markdown + PDF export
├── data/samples/            # Pre-loaded JD + resume pairs
├── .streamlit/config.toml
├── .env.example
├── requirements.txt
└── README.md
```

## 🎬 Demo

_(Add your demo video link here once recorded)_

## 🎯 Proficiency Rubric

Every skill is scored 1–5:

| Score | Label | Criteria |
|-------|-------|----------|
| 1 | Novice | No demonstrated knowledge |
| 2 | Basic Awareness | Recognises terminology but can't apply |
| 3 | Working Knowledge | Can explain and has applied in limited contexts |
| 4 | Strong Proficiency | Deep understanding, solves complex problems |
| 5 | Expert Mastery | Can teach others, architect solutions, handle edge cases |

## 📝 License

Built for hackathon submission. MIT-friendly — use the code however you like.
