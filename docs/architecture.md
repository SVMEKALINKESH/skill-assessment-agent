# Architecture

## System Overview

```mermaid
graph TD
    subgraph User[👤 Candidate]
        Browser[Web Browser]
    end

    subgraph Deploy[☁️ Streamlit Cloud]
        App[app.py<br/>Streamlit UI]
    end

    subgraph Core[🧠 Core Pipeline]
        Pipeline[pipeline.py<br/>Orchestrator]
        Prompts[agent_prompts.py<br/>System Prompts]
        Parser[resume_parser.py<br/>PDF/DOCX/TXT]
        Exporter[exporter.py<br/>PDF/Markdown]
        LLM[llm_provider.py<br/>Gemini / OpenAI / ...]
    end

    subgraph External[🌐 External]
        Gemini[Google Gemini API<br/>Free Tier]
        GitHub[(GitHub Repo)]
    end

    Browser <-->|HTTPS| App
    App --> Parser
    App --> Pipeline
    App --> Exporter
    Pipeline --> Prompts
    Pipeline --> LLM
    LLM -->|REST| Gemini
    GitHub -->|auto-deploy on push| Deploy

    classDef ui fill:#4F46E5,stroke:#312E81,color:#fff
    classDef core fill:#10B981,stroke:#065F46,color:#fff
    classDef ext fill:#F59E0B,stroke:#92400E,color:#fff
    class Browser,App ui
    class Pipeline,Prompts,Parser,Exporter,LLM core
    class Gemini,GitHub ext
```

## The 7-Step Agent Flow

```mermaid
flowchart TD
    Start([Candidate: upload resume + paste JD]) --> S1
    S1["Step 1: Skill Analysis<br/>🤖 LLM extracts + matches skills<br/>Experience year weighting"] --> S2
    S2["Step 2: Match Percentage<br/>🤖 LLM calculates weighted match %"] --> S3
    S3{"Step 3: Gap Classification<br/>⚙️ Code: Low / Medium / High"}
    S3 --> S4
    S4["Step 4: Conversational Assessment<br/>🤖 LLM: 1-3 questions per skill<br/>Adaptive difficulty"]
    S4 --> |"loop per skill"| S4
    S4 --> S5["Step 5: Revised Gap Analysis<br/>🤖 LLM revises scores with<br/>real assessment data"]
    S5 --> S6["Step 6: Learning Plan Generation<br/>🤖 LLM: resources, time estimates,<br/>adjacent skills"]
    S6 --> S7["Step 7: Display + Export<br/>⚙️ Streamlit UI<br/>PDF / Markdown download"]
    S7 --> End([Candidate gets report])

    classDef llm fill:#10B981,stroke:#065F46,color:#fff
    classDef code fill:#4F46E5,stroke:#312E81,color:#fff
    class S1,S2,S4,S5,S6 llm
    class S3,S7 code
```

🤖 = LLM-powered step   ⚙️ = Deterministic code

## Scoring Logic

**Proficiency rubric (1-5):**

| Score | Label | Meaning |
|-------|-------|---------|
| 1 | Novice | No demonstrated knowledge |
| 2 | Basic Awareness | Recognises terminology, can't apply |
| 3 | Working Knowledge | Can explain and has applied in limited contexts |
| 4 | Strong Proficiency | Deep understanding, solves complex problems |
| 5 | Expert Mastery | Can teach others, architect solutions |

**Experience year weighting (Step 1):**

- Resume years == JD required years → full score (1.0)
- Resume years ≥ JD required + 1 → bonus (1.1)
- Resume years < JD required → proportional (years_have / years_required)
- No year info → score based on contextual evidence only

**Gap classification (Step 3):**

- Low gap: match < 50%  🔴
- Medium gap: 50% ≤ match ≤ 75%  🟡
- High gap: match > 75%  🟢

**Gap per skill (Step 5):**

- gap_size = max(0, JD_required - revised_score)
- 0 → "no_gap"
- 1 → "minor"
- 2 → "moderate"
- 3+ → "significant"

## Data Flow (per session)

All state lives in `st.session_state` — no database, no persistence between
sessions. The user's resume + JD never leave their browser session except to
make direct calls to the Gemini API.

```
st.session_state = {
  "step": "input" | "skill_overview" | "assessment" | "learning_plan",
  "jd_text": str,
  "resume_text": str,
  "skill_analysis": dict,           # Step 1 output
  "match_percentage": dict,          # Step 2 output
  "gap_classification": str,         # Step 3 output
  "assessment_state": dict,          # Step 4 running state
  "revised_analysis": dict,          # Step 5 output
  "learning_plan": dict,             # Step 6 output
  "api_key": str,
}
```
