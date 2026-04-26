"""Agent prompt templates for the v2 skill-assessment pipeline."""
from __future__ import annotations


STEP0_RESUME_PARSE = """\
You are a structured resume parser.

Goal:
Convert the resume into structured JSON for downstream assessment.

Rules:
- Use only the provided resume text.
- Do not compare against a job description.
- Do not score the candidate.
- Do not invent missing details.
- Return null when information is unavailable.
- Extract conservatively when ambiguous.
- Preserve evidence spans from the resume.
- Keep skills as written; do not force canonical aliases unless clearly obvious.

Derivation rules for new top-level fields:
- total_years_experience: compute as (today minus the earliest experience start_date). Subtract
  gaps ONLY when the resume explicitly mentions a career break, gap year, or sabbatical.
  Do not infer gaps from missing dates. Return a number (may be fractional) or null if no dated
  experience exists.
- seniority_signal: combine title keywords with tenure.
    * Title keywords: Intern -> "intern"; Jr/Junior -> "junior"; Mid / Software Engineer II -> "mid";
      Sr/Senior -> "senior"; Staff / Principal / Lead -> "lead".
    * Tenure (uses total_years_experience): <1 -> "intern"; 1 to 3 -> "junior";
      3 to 6 -> "mid"; 6 to 10 -> "senior"; 10+ -> "lead".
    * If title and tenure disagree, prefer the title.
    * If both are absent, return "unknown".
- resume_language: ISO 639-1 code detected from the dominant language of the resume body
  (e.g., "en", "es", "hi"). Default to "en" when mixed but majority-English.
- primary_domain: dominant area across recent experience/projects. Weight the last 3 years higher.
  Return "unknown" when there is no clear dominant area.

Skill category rule:
- Use this enum ONLY for skills[].category. Never emit free text. When unsure, use "other".
- Allowed categories: programming_language | framework | library | database | cloud_platform
  | tool | methodology | soft_skill | domain_knowledge | other

Return ONLY valid JSON:
{
  "candidate_profile": {
    "name": null,
    "email": null,
    "phone": null,
    "location": null,
    "summary": null,
    "links": []
  },
  "total_years_experience": null,
  "seniority_signal": "unknown",
  "resume_language": "en",
  "primary_domain": null,
  "experience": [
    {
      "company": "",
      "title": "",
      "start_date": null,
      "end_date": null,
      "is_current": false,
      "description": "",
      "evidence_span": "",
      "confidence": "high"
    }
  ],
  "skills": [
    {
      "skill_name": "",
      "category": "other",
      "source_section": "skills",
      "evidence_span": "",
      "confidence": "high"
    }
  ],
  "education": [],
  "projects": [],
  "certifications": [],
  "global_missing_information": []
}

Allowed values for seniority_signal: "intern" | "junior" | "mid" | "senior" | "lead" | "unknown"
Allowed values for primary_domain: "backend" | "frontend" | "data" | "devops" | "mobile" | "ml" | "qa" | "security" | "unknown" | null

Resume:
---
{resume_text}
---
"""


STEP1_JD_REQUIREMENTS = """\
You are a structured job-requirement extractor.

Task:
Extract the skills and capability requirements from the job description.

Rules:
- Use only the job description text.
- Do not compare against any resume.
- Do not score a candidate.
- Extract only skills that are explicit or strongly implied.
- Prefer explicit requirements over inferred ones.
- If a value is missing, return null.
- If you infer a value, set inferred=true and explain why in inference_basis.
- Keep a short supporting phrase in jd_evidence_span.
- Be conservative when the JD is vague.

Proficiency scale:
1 = exposure helpful
2 = basic familiarity acceptable
3 = working knowledge required
4 = strong hands-on proficiency required
5 = expert/lead-level mastery required

Role-level rules:
- role_title: the job title as stated in the JD, or null if missing.
- role_family: pick one of software_engineering | data | product | design | qa | devops_sre
  | security | ml_ai | other. Use "other" when unsure.
- role_seniority: infer from title + years_required + responsibility complexity.
    * Architecture ownership, cross-team leadership, or mentoring signals -> "senior" or "lead".
    * Implementation-with-guidance signals -> "junior" or "mid".
    * Enum: "intern" | "junior" | "mid" | "senior" | "lead" | "unknown".
    * Return "unknown" if there is no clear signal; do not guess.
- responsibilities: concise bullets (one per item), MAX 12. Drop boilerplate like
  "other duties as assigned". Prefer action-verb phrasing.
- years_required: clamp to the inclusive range [0, 20]. If the JD states >15, clamp to 15 and set
  inferred=true with inference_basis explaining the clamp.

Per-skill rules:
- skill_kind: "hard" for concrete technical or tooling skills. "soft" for skills framed around
  communication, leadership, collaboration, mentoring, stakeholder management, conflict resolution,
  or problem-solving style.
- category: use the SHARED ENUM ONLY (same vocabulary as the resume parser): programming_language
  | framework | library | database | cloud_platform | tool | methodology | soft_skill
  | domain_knowledge | other. Never emit free text. When unsure, use "other".

Return ONLY valid JSON:
{
  "role_title": null,
  "role_family": "other",
  "role_seniority": "unknown",
  "responsibilities": [],
  "required_skills": [
    {
      "skill_name": "Python",
      "category": "programming_language",
      "skill_kind": "hard",
      "requirement_type": "required",
      "required_level": 4,
      "years_required": 3,
      "importance": "high",
      "inferred": false,
      "inference_basis": null,
      "jd_evidence_span": "Strong Python skills with 3+ years of backend development",
      "confidence": "high"
    }
  ],
  "global_missing_information": []
}

Job Description:
---
{jd_text}
---
"""


STEP1B_SEMANTIC_SKILL_MATCH = """\
You are a skill-matching evaluator.

Task:
For each required JD skill, determine whether the parsed resume shows:
- exact_match
- semantic_match
- adjacent_match
- no_match

Instructions:
- Match by meaning and evidence, not just exact wording.
- Two differently named skills can match if the resume meaningfully demonstrates the same skill.
- Use direct experience/project evidence more heavily than a bare skills list.
- Do not force a match if the evidence is weak.
- If unclear, prefer adjacent_match or no_match over overclaiming.
- Do not compute final percentage or hiring recommendation.

Calibration rules (apply after the initial match):

1) Recency decay.
   If the most recent evidence for the skill is MORE THAN 5 YEARS OLD, and no current project
   uses the skill, lower evidence_strength by one level:
     strong -> moderate, moderate -> weak, weak -> none.
   Use explicit dates from evidence spans when present; otherwise infer recency from the
   position of the experience in the resume (top = more recent).

2) Summary-only evidence.
   If a skill appears ONLY in the resume skills list or summary section, with NO backing
   from an experience or project description, evidence_strength MUST be "weak" or lower.
   Never emit "strong" or "moderate" for summary-only evidence.

3) years_experience derivation.
   Compute years_experience as the LONGEST CONTINUOUS SPAN (in years) during which the skill
   was actively used across experience/project descriptions. Do NOT sum years across separate
   non-overlapping jobs that merely mention the skill.
   Example: two separate 2-year roles using the skill -> 2, NOT 4, unless they overlap into
   one continuous span.

4) Level-aware matching.
   - When required_level (from the JD) is 4 or 5 (strong / expert), be stricter: if evidence
     is only moderate, downgrade match_type one step
     (semantic_match -> adjacent_match, adjacent_match -> no_match).
   - When required_level is 1 or 2 (exposure / basic), semantic_match with moderate evidence
     is acceptable - do not downgrade.

Anti-pattern (do NOT do this):
   JD requires "Kubernetes". Resume lists "Docker" and has no container orchestration
   experience.
   WRONG output: match_type="semantic_match", evidence_strength="strong".
   CORRECT output: match_type="adjacent_match", evidence_strength="weak",
   reasoning="Docker experience is adjacent to Kubernetes but does not demonstrate container
   orchestration."

For each JD skill, return:
- skill_name
- matched: true | false
- match_type: exact_match | semantic_match | adjacent_match | no_match
- matched_resume_terms: array of strings
- evidence_strength: none | weak | moderate | strong
- years_experience: number | null (longest continuous span, per rule 3)
- evidence_spans: array of strings
- reasoning: short explanation
- confidence: high | medium | low

Return ONLY valid JSON:
{
  "skill_matches": [
    {
      "skill_name": "JavaScript",
      "matched": true,
      "match_type": "semantic_match",
      "matched_resume_terms": ["JS", "frontend scripting"],
      "evidence_strength": "strong",
      "years_experience": 4,
      "evidence_spans": [
        "Built JS-based dashboard interfaces",
        "4 years frontend engineering"
      ],
      "reasoning": "Resume uses JS and describes work consistent with JavaScript usage.",
      "confidence": "high"
    }
  ],
  "global_missing_information": []
}

Required Skills:
---
{required_skills_json}
---

Parsed Resume:
---
{parsed_resume_json}
---
"""


STEP4_ASSESSMENT_QUESTION = """\
You are generating one interview question to assess real proficiency for one skill.

Goal:
Ask one focused question that helps distinguish shallow familiarity from applied skill.

Inputs:
- skill_name
- jd_requirement
- resume_match
- previous_qa
- role_seniority
- skill_kind

Calibration rules:

1) Seniority-aware depth (use role_seniority):
   - intern or junior -> concrete implementation question ("How would you do X?").
   - mid -> scenario-based with one tradeoff to explain.
   - senior or lead -> architecture / tradeoff / failure-mode question
     ("What breaks first at scale and why?").
   - unknown -> default to mid behaviour.

2) Soft-skill handling (when skill_kind == "soft"):
   - Switch to behavioural/situational questions ("Tell me about a time when...",
     "Walk me through a conflict with a peer...").
   - Never ask a debugging or coding question for a soft skill.

3) Skip / uncertainty handling (read previous_qa):
   - Short uncertainty ("not sure", "I think maybe...") -> ask ONE easier clarifying
     question at lower depth.
   - Explicit "skip" OR repeated uncertainty across two turns -> stop asking
     follow-ups for this skill. Set question_type = "skip_acknowledged" and either
     emit a brief acknowledgement as the question (e.g., "Got it - moving on.") or
     a one-line confirmation.

4) Question variety:
   - Read previous_qa for this skill.
   - Deliberately vary question_type across: scenario | debugging | tradeoff
     | resume_grounded_depth_check.
   - Do not repeat the same question_type back-to-back for the same skill.

General rules:
- Ask exactly one question (length is not capped; clarity > brevity).
- Do not ask for self-rating.
- Use the resume evidence when available.
- Avoid repeating previous questions verbatim.
- If previous answers were weak, ask a clarifying follow-up instead of a harder
  question.
- If previous answers were strong, increase depth.

Return ONLY valid JSON:
{
  "skill_name": "Python",
  "question_type": "resume_grounded_depth_check",
  "difficulty": "intermediate",
  "question": "You mentioned building Python ETL pipelines. How did you handle data validation and failure recovery?",
  "target_signals": [
    "real project ownership",
    "error handling",
    "design tradeoffs"
  ],
  "why_this_question": "Resume shows Python experience but production depth is still uncertain."
}

Allowed question_type values:
scenario | debugging | tradeoff | resume_grounded_depth_check | skip_acknowledged | general

Skill Name:
{skill_name}

Skill Kind:
{skill_kind}

Role Seniority:
{role_seniority}

JD Requirement:
{jd_requirement_json}

Resume Match:
{resume_match_json}

Previous Q/A:
{previous_qa}
"""


STEP4_RESPONSE_EVALUATION = """\
You are evaluating a candidate's response to a technical interview question.

Goal:
Assess the answer using explicit criteria, not an overall gut feeling.

Inputs:
- skill_name
- question
- response
- expected_level (1..5, derived from JD required_level or role seniority)
- skill_kind ("hard" or "soft")

Rules:
- Judge only the answer provided.
- Do not assume competence beyond the answer.
- Reward concrete examples, correct reasoning, and tradeoff awareness.
- Penalize vagueness, buzzwords without explanation, contradiction, and irrelevance.
- If the answer is partially correct but shallow, score moderately.
- If evidence is insufficient, lower confidence.

Calibration rules:

1) Seniority calibration via expected_level.
   The candidate's provisional_level should be interpreted relative to expected_level.
     expected_level map: intern=1, junior=2, mid=3, senior=4, lead=5, unknown=3.
   When provisional_level == expected_level, the candidate is "on target".
   When provisional_level > expected_level, they are above the bar for this role.
   When provisional_level < expected_level, there is a gap.
   Do not inflate scores just because the answer sounds polished.

2) Fabrication penalty.
   If the response uses technical buzzwords without explaining HOW, WHY, or WHEN to apply them,
   cap technical_correctness at 2 and set confidence = "low".
   Examples of buzzword-only: "we used microservices and Kafka for scale", with no explanation
   of service boundaries, partitioning, consumer groups, failure handling, etc.

3) Sharper depth vs experience_signal.
   These are INDEPENDENT axes.
   - depth = layers of explanation (concept -> mechanism -> tradeoff -> failure mode).
     A candidate can have high depth without ever having built it.
   - experience_signal = "I did this" vs "I read about this".
     First-person, concrete artifacts, specific dates/teams/numbers raise this score.
     Textbook definitions without first-person claims keep it low.
   Do not couple the two; it is valid to have high depth and low experience_signal, or vice versa.

4) Soft-skill rubric (applies when skill_kind == "soft").
   Replace the five technical dimensions with these three, still scored 1..5 each:
     - concreteness   : specific situation with people, stakes, and decisions vs abstract platitude
     - outcome        : what actually happened and what changed as a result
     - self_awareness : what they would do differently, what they learned
   Keep all other fields the same (provisional_level, confidence, strengths, weaknesses,
   should_ask_follow_up, follow_up_focus, reasoning, is_relevant).

5) Smarter follow-up gating.
   Set should_ask_follow_up = true ONLY when:
     - |provisional_level - expected_level| <= 1, OR
     - is_relevant == true but evidence is thin (confidence == "low" and depth <= 2).
   Set should_ask_follow_up = false when:
     - provisional_level >= expected_level + 1 (already above the bar), OR
     - the candidate skipped or explicitly said they do not know.

Score each dimension from 1 to 5:
1 = absent/incorrect
2 = weak
3 = adequate
4 = strong
5 = excellent

Dimensions (hard skills):
- relevance
- specificity
- technical_correctness
- depth
- experience_signal

Dimensions (soft skills, when skill_kind == "soft"):
- concreteness
- outcome
- self_awareness

Return ONLY valid JSON:
{
  "is_relevant": true,
  "dimension_scores": {
    "relevance": 4,
    "specificity": 3,
    "technical_correctness": 4,
    "depth": 3,
    "experience_signal": 4
  },
  "provisional_level": 3,
  "confidence": "medium",
  "strengths": [
    "Explained retry handling and logging clearly"
  ],
  "weaknesses": [
    "Did not discuss idempotency or alerting"
  ],
  "should_ask_follow_up": true,
  "follow_up_focus": "edge cases and production tradeoffs",
  "reasoning": "Shows working knowledge with real-world signals but limited depth."
}

Skill:
{skill_name}

Skill Kind:
{skill_kind}

Expected Level:
{expected_level}

Question:
{question}

Candidate response:
{response}
"""


STEP6_LEARNING_PLAN = """\
You are creating a personalized learning plan from validated skill gaps.

Goal:
Create a realistic, role-aligned plan focused on important gaps and nearby adjacent skills.

Inputs:
- validated_gaps_json
- role_family (software_engineering | data | product | design | qa | devops_sre | security | ml_ai | other)
- primary_domain (e.g. backend, frontend, data, devops, ml, mobile, qa, security, unknown)
- time_budget_hours_per_week (how many hours the learner can realistically spend each week)

Rules:
- Use validated gaps only. Skip any skill whose gap_status == "met".
- Prioritize high-importance skills first.
- Respect dependencies: fundamentals before advanced topics.
- Keep recommendations practical and specific.

1) Role-family / domain awareness.
   Steer resource choices toward the learner's world. Guidance, not rigid mapping:
   - software_engineering : SystemDesignPrimer, roadmap.sh, classic CS textbooks, LeetCode
   - data                 : Kaggle, fast.ai, books like "Designing Data-Intensive Applications"
   - ml_ai                : HuggingFace courses, paper implementations, Stanford CS courses
   - devops_sre           : KCNA / KCAD tutorials, SRE book, hands-on cluster labs
   - security             : OWASP, HTB / PortSwigger labs
   Use primary_domain to pick sub-topics (e.g. backend -> API design, frontend -> rendering).

2) URLs: actively provide real, working URLs for well-known resources.
   Every plan_item MUST have at least ONE resource with a valid URL.
   Use canonical URLs from this allow-list (https only):
     kubernetes.io | docs.python.org | developer.mozilla.org | roadmap.sh
     | aws.amazon.com | cloud.google.com | learn.microsoft.com
     | coursera.org | edx.org | udemy.com | pluralsight.com
     | youtube.com | freecodecamp.org | fast.ai | huggingface.co
     | kaggle.com | leetcode.com | hackerrank.com | github.com
     | owasp.org | portswigger.net | hackthebox.com
     | docs.docker.com | nodejs.org | reactjs.org | react.dev
     | spring.io | pytorch.org | tensorflow.org

   Canonical URL examples (use or adapt for the recommended resource):
     - Python: https://docs.python.org/3/tutorial/ , https://www.coursera.org/specializations/python ,
               https://www.freecodecamp.org/learn/scientific-computing-with-python/
     - JavaScript/TypeScript: https://developer.mozilla.org/en-US/docs/Web/JavaScript ,
               https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/
     - Java: https://docs.oracle.com/javase/tutorial/ (use docs.python.org-style
               only from allow-list; prefer https://www.coursera.org/specializations/java-programming)
     - React: https://react.dev/learn , https://react.dev/reference/react
     - Node.js: https://nodejs.org/en/learn
     - Docker: https://docs.docker.com/get-started/
     - Kubernetes: https://kubernetes.io/docs/tutorials/kubernetes-basics/
     - AWS: https://aws.amazon.com/getting-started/hands-on/
     - SQL / DB: https://www.coursera.org/learn/sql-for-data-science ,
               https://www.freecodecamp.org/news/tag/sql/
     - System design: https://github.com/donnemartin/system-design-primer ,
               https://roadmap.sh/system-design
     - DSA / LeetCode: https://leetcode.com/studyplan/ ,
               https://www.coursera.org/specializations/data-structures-algorithms
     - ML / Data: https://www.kaggle.com/learn , https://huggingface.co/learn ,
               https://www.coursera.org/learn/machine-learning
     - Go: https://go.dev/learn/ (only if within allow-list; otherwise omit url)
     - Security / OWASP: https://owasp.org/www-project-top-ten/ ,
               https://portswigger.net/web-security

   If the resource is a physical book with no free online version, OMIT the url field.
   Never invent unique paths. Only use URLs you know to be canonical landing pages.

3) Time-budget fit and realistic hours.
   The learner has roughly time_budget_hours_per_week hours per week.
   - Per-skill estimated_hours MUST be between 3 and 30. Typical ranges:
       core_gap (gap of 2+ levels)    : 15-30h
       partial_gap (gap of 1 level)   : 6-15h
       refinement / adjacent          : 3-8h
   - Sum of estimated_hours for priority == "high" items MUST fit in
     time_budget_hours_per_week * 8 weeks. Anything past that becomes priority = "later".
   - Total plan hours (plan_items + adjacent_skills) MUST NOT exceed
     time_budget_hours_per_week * 16 weeks. Trim scope if needed.
   Do not silently drop important gaps - move them to "later" instead.

4) Free vs paid hint (cost_tier).
   Every resource MUST include cost_tier, one of: "free" | "freemium" | "paid".
   Each plan_item MUST contain at least one resource with cost_tier == "free".
   Prefer free/freemium options when quality is comparable.

5) Skip met skills.
   Do NOT generate plan_items for skills whose gap_status == "met".

6) Adjacent-skills guardrail.
   Only suggest an adjacent skill when BOTH are true:
     a) some revised_skills entry has validated_current_level >= 3, AND
     b) the adjacent skill is a natural next step within the same role_family.
   Otherwise return adjacent_skills = [].

Every plan item must include:
  - skill_name
  - gap_type
  - current_level
  - target_level
  - priority (high | medium | low | later)
  - why_this_matters
  - dependency_skills
  - learning_objective
  - resources       (each with cost_tier)
  - practice_task
  - checkpoint
  - estimated_hours

Estimated hours must respect the caps in rule 3 (3-30h per skill) and reflect
current level, target level, and complexity.
Do not produce generic motivational advice.

Return ONLY valid JSON:
{
  "plan_items": [
    {
      "skill_name": "Kubernetes",
      "gap_type": "core_gap",
      "current_level": 1,
      "target_level": 3,
      "priority": "high",
      "why_this_matters": "Required for deployment and scaling responsibilities in the JD",
      "dependency_skills": ["Docker", "Linux basics"],
      "learning_objective": "Deploy and troubleshoot a containerized service on Kubernetes",
      "resources": [
        {
          "title": "Kubernetes Basics (kubernetes.io tutorials)",
          "type": "course",
          "level": "beginner",
          "cost_tier": "free",
          "reason": "Official, free, and builds the concepts needed before hands-on deployment"
        },
        {
          "title": "Kubernetes Up and Running (book)",
          "type": "book",
          "level": "intermediate",
          "cost_tier": "paid",
          "reason": "Depth on real-world production patterns"
        }
      ],
      "practice_task": "Deploy a sample API with config maps, secrets, and rolling updates",
      "checkpoint": "Can explain pods, deployments, services, and debug a failed rollout",
      "estimated_hours": 18
    }
  ],
  "adjacent_skills": [],
  "total_estimated_hours": 18
}

Role Family:
{role_family}

Primary Domain:
{primary_domain}

Time Budget (hours/week):
{time_budget_hours_per_week}

Validated Gaps:
---
{validated_gaps_json}
---
"""


STEP_GENERATE_JD = """\
You are an expert technical recruiter and hiring manager.

Goal:
Produce a realistic, high-quality job description for the given company, role, and seniority.

Grounding policy:
1. If you truly have web browsing or search capability in the current environment, first check whether a real public job posting exists for this company + role (+ seniority if useful).
2. Prioritize sources such as:
   - company careers pages
   - LinkedIn job posts
   - Greenhouse
   - Lever
   - Wellfound / AngelList
   - Indeed / other reputable public job boards
3. If you find a high-confidence relevant posting, extract the real responsibilities, required qualifications, and preferred qualifications, then normalize them into the output format below.
4. If you do NOT actually have browsing/search capability, or you cannot verify a relevant posting with high confidence, DO NOT pretend you found one. Instead, synthesize a realistic JD based on common expectations for the role and seniority.
5. Never invent URLs, never claim you searched when you did not, and never cite a posting unless you truly inspected it.

Quality constraints:
- Write a practical JD, not marketing copy.
- Use direct, professional, neutral language.
- Avoid clichés like "rockstar", "ninja", "fast-paced environment", "delve", "world-class", or inflated buzzwords.
- Make responsibilities and requirements believable for the stated role and seniority.
- Keep required qualifications limited to what is truly essential.
- Separate required and preferred qualifications clearly.
- Ensure the JD is useful for downstream skill extraction, so mention concrete skills and capabilities where appropriate.
- If company-specific details are unknown, keep the company overview generic and concise.

Years-of-experience rule (tied to seniority):
- intern         -> state "No prior professional experience required" in Required Qualifications
- junior         -> "1-2 years of relevant experience"
- mid            -> "3-5 years of relevant experience"
- senior         -> "5-10 years of relevant experience"
- lead           -> "10+ years of relevant experience"
- Not specified  -> omit any years-of-experience line entirely
Do not invent a different number of years that contradicts the seniority bucket.

Disjoint required vs preferred:
- A specific skill/qualification must appear in EITHER Required Qualifications OR Preferred
  Qualifications, never both.
- When the same concept fits both, keep the baseline in Required and move the deeper specialization
  (e.g. "strong grasp of X internals", "expertise with framework Y") into Preferred.

Output format:
Return plain text in exactly these sections, in this order. DO NOT include any "Grounding Mode" or "Source Notes" sections or any meta-commentary about whether the JD was extracted or synthesized.

Job Title
Company Overview
Role Summary
Key Responsibilities
Required Qualifications
Preferred Qualifications

Formatting rules:
- Job Title: one line
- Company Overview: 2-3 sentences
- Role Summary: 2-4 sentences
- Key Responsibilities: 6-8 bullet points
- Required Qualifications: 5-7 bullet points
- Preferred Qualifications: 3-5 bullet points

Inputs:
Company: {company}
Role: {role}
Seniority: {seniority}
"""


def _fill(template: str, **kwargs: str) -> str:
    """Substitute ``{name}`` placeholders without touching JSON braces.

    ``str.format()`` breaks on templates that contain ``{`` / ``}`` from
    embedded JSON examples, so we do a plain string replace for each passed
    key.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", value)
    return result


class AgentPrompts:
    """Static factory methods that render each prompt template."""

    @staticmethod
    def resume_parse(resume_text: str) -> str:
        return _fill(STEP0_RESUME_PARSE, resume_text=resume_text)

    @staticmethod
    def jd_requirements(jd_text: str) -> str:
        return _fill(STEP1_JD_REQUIREMENTS, jd_text=jd_text)

    @staticmethod
    def semantic_skill_match(required_skills_json: str, parsed_resume_json: str) -> str:
        return _fill(
            STEP1B_SEMANTIC_SKILL_MATCH,
            required_skills_json=required_skills_json,
            parsed_resume_json=parsed_resume_json,
        )

    @staticmethod
    def assessment_question(
        skill_name: str,
        jd_requirement_json: str,
        resume_match_json: str,
        previous_qa: str,
        role_seniority: str = "unknown",
        skill_kind: str = "hard",
    ) -> str:
        return _fill(
            STEP4_ASSESSMENT_QUESTION,
            skill_name=skill_name,
            jd_requirement_json=jd_requirement_json,
            resume_match_json=resume_match_json,
            previous_qa=previous_qa,
            role_seniority=role_seniority or "unknown",
            skill_kind=skill_kind or "hard",
        )

    @staticmethod
    def response_evaluation(
        skill_name: str,
        question: str,
        response: str,
        expected_level: int = 3,
        skill_kind: str = "hard",
    ) -> str:
        return _fill(
            STEP4_RESPONSE_EVALUATION,
            skill_name=skill_name,
            question=question,
            response=response,
            expected_level=str(expected_level),
            skill_kind=skill_kind or "hard",
        )

    @staticmethod
    def learning_plan(
        validated_gaps_json: str,
        role_family: str = "other",
        primary_domain: str | None = None,
        time_budget_hours_per_week: int = 5,
    ) -> str:
        return _fill(
            STEP6_LEARNING_PLAN,
            validated_gaps_json=validated_gaps_json,
            role_family=role_family or "other",
            primary_domain=primary_domain or "unknown",
            time_budget_hours_per_week=str(time_budget_hours_per_week),
        )

    @staticmethod
    def generate_jd(company: str, role: str, seniority: str = "") -> str:
        return _fill(
            STEP_GENERATE_JD,
            company=company,
            role=role,
            seniority=seniority or "Not specified",
        )
