"""
AI Recruitment System
Merges:
 - PDF Resume Ingestion (RAG via LangChain + Chroma + HuggingFace)
 - AI Interview Bot (sequential Q&A with session state)
 - Candidate Screening & Scoring Pipeline (match score, skills gap, decision)
LLM: Ollama (local) — default model: llama3.2
"""

import os
import json
import tempfile
import uuid
import streamlit as st
from dotenv import load_dotenv

# LangChain / PDF stack
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "tinyllama")
EMBED_MODEL     = "all-MiniLM-L6-v2"
MATCH_THRESHOLD = 70   # % minimum to proceed to interview
NUM_QUESTIONS   = 5

# ──────────────────────────────────────────────
# PROMPTS  (from your config.py, extended)
# ──────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────
SCREENING_PROMPT = """Output ONLY a JSON object. No explanation. No markdown.

Resume: {resume_text}
Job: {job_description}
Candidate: {name}, {experience} yrs exp, role: {current_role}, edu: {education}

Return this JSON (fill real values):
{{"extracted_skills":["skill1"],"years_experience":"{experience}","education_summary":"{education}","match_score":70,"skills_score":70,"experience_score":70,"projects_score":70,"education_score":70,"decision":"Proceed","matching_skills":["skill1"],"missing_skills":["skill2"],"strengths":["strength1"],"weaknesses":["weakness1"],"reject_reason":""}}

Rule: decision=Proceed if match_score>={threshold} else Reject.
Output JSON only:"""

QUESTIONS_PROMPT = """Output ONLY a numbered list of {num_questions} interview questions. No preamble.

Role: {job_description}
Candidate has: {candidate_skills}
Candidate lacks: {missing_skills}

1."""

EVALUATION_PROMPT = """Output ONLY a JSON object. No markdown. No explanation.

Transcript: {interview_text}
Job: {job_description}

Return this JSON (fill real values):
{{"technical_score":70,"problem_solving_score":70,"communication_score":70,"overall_score":70,"strengths":["strength1"],"weak_areas":["weakness1"],"recommendation":"Hold","summary":"Brief assessment here."}}

recommendation must be Hire, Hold, or Reject.
Output JSON only:"""

ANSWER_EVAL_PROMPT = """Output ONLY a JSON object. No markdown. No explanation.

Question: {question}
Answer: {answer}

Return this JSON (fill real values):
{{"feedback":"One sentence feedback here.","score":6}}

Output JSON only:"""

# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
def init_session():
    defaults = {
        "stage": "details",           # details → resume → screening → interview → result
        "candidate": {},
        "jd": "",
        "resume_text": "",
        "resume_chunks": None,
        "vector_db": None,
        "screening_result": None,
        "questions": [],
        "answers": [],
        "answer_feedback": [],
        "interview_step": 0,
        "final_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def get_llm():
    return ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0)

def ingest_pdf(uploaded_file) -> tuple[str, list]:
    """Load PDF → return (full_text, chunks)"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)
    full_text = "\n".join(d.page_content for d in docs)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    return full_text, chunks

@st.cache_resource
def build_vector_db(_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    return Chroma.from_documents(documents=_chunks, embedding=embeddings)

def call_llm_json(prompt: str) -> dict:
    """Call Ollama and robustly extract JSON from response."""
    import re
    llm = get_llm()
    raw = llm.invoke(prompt).content

    # 1. Strip markdown fences
    clean = raw.strip()
    clean = re.sub(r"```json|```", "", clean).strip()

    # 2. Try direct parse first
    try:
        result = json.loads(clean)
        # tinyllama sometimes wraps the object in a list
        if isinstance(result, list):
            result = result[0] if result and isinstance(result[0], dict) else {}
        return result
    except json.JSONDecodeError:
        pass

    # 3. Extract first {...} block (handles tinyllama adding extra text)
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                result = result[0] if result and isinstance(result[0], dict) else {}
            return result
        except json.JSONDecodeError:
            pass

    # 4. Last resort: return a safe default so the app doesn't crash
    st.warning("⚠️ Model returned unexpected output — using fallback scores.")
    return {
        "extracted_skills": [], "years_experience": "N/A",
        "education_summary": "N/A", "match_score": 60,
        "skills_score": 60, "experience_score": 60,
        "projects_score": 60, "education_score": 60,
        "decision": "Reject", "matching_skills": [],
        "missing_skills": [], "strengths": ["Unable to parse"],
        "weaknesses": ["Model output was not valid JSON"],
        "reject_reason": "Could not evaluate — try a larger model.",
        "technical_score": 60, "problem_solving_score": 60,
        "communication_score": 60, "overall_score": 60,
        "recommendation": "Hold", "summary": "Evaluation incomplete.",
        "weak_areas": [], "feedback": "N/A", "score": 5,
    }

def parse_questions(raw: str) -> list[str]:
    """Extract numbered questions from raw LLM text."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    questions = []
    for line in lines:
        # Remove numbering like "1.", "1)", "Q1."
        for prefix in [f"{i}." for i in range(1, 20)] + [f"{i})" for i in range(1, 20)]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            questions.append(line)
    return questions[:NUM_QUESTIONS]

def score_color(score: int) -> str:
    if score >= 75:
        return "🟢"
    elif score >= 50:
        return "🟡"
    return "🔴"

def stage_index(stage: str) -> int:
    return ["details", "resume", "screening", "interview", "result"].index(stage)

# ──────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────
def render_progress():
    stages = ["Details", "Resume", "Screening", "Interview", "Result"]
    current = stage_index(st.session_state.stage)
    cols = st.columns(len(stages))
    for i, (col, label) in enumerate(zip(cols, stages)):
        if i < current:
            col.markdown(f"<div style='text-align:center;color:#22c55e;font-size:13px;font-weight:600'>✓ {label}</div>", unsafe_allow_html=True)
        elif i == current:
            col.markdown(f"<div style='text-align:center;font-size:13px;font-weight:700;border-bottom:2px solid #6366f1;padding-bottom:4px'>{label}</div>", unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align:center;color:#9ca3af;font-size:13px'>{label}</div>", unsafe_allow_html=True)
    st.divider()

# ──────────────────────────────────────────────
# STAGE 1 — CANDIDATE DETAILS
# ──────────────────────────────────────────────
def stage_details():
    st.subheader("Candidate Details")

    with st.form("details_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Full Name *", placeholder="e.g. Priya Sharma")
        email = c2.text_input("Email *", placeholder="e.g. priya@email.com")

        c3, c4 = st.columns(2)
        experience = c3.text_input("Years of Experience", placeholder="e.g. 4")
        current_role = c4.text_input("Current Role", placeholder="e.g. Frontend Developer")

        education = st.text_input("Education", placeholder="e.g. B.Tech Computer Science, IIT Delhi")

        st.markdown("##### Job Description *")
        jd = st.text_area(
            "Paste the full job description",
            height=200,
            placeholder="Include required skills, responsibilities, qualifications...",
            label_visibility="collapsed"
        )

        submitted = st.form_submit_button("Continue →", type="primary", use_container_width=True)

    if submitted:
        if not name.strip() or not jd.strip():
            st.error("Please fill in your name and the job description.")
            return
        st.session_state.candidate = {
            "name": name.strip(),
            "email": email.strip(),
            "experience": experience.strip(),
            "current_role": current_role.strip(),
            "education": education.strip(),
        }
        st.session_state.jd = jd.strip()
        st.session_state.stage = "resume"
        st.rerun()

# ──────────────────────────────────────────────
# STAGE 2 — RESUME UPLOAD (PDF RAG)
# ──────────────────────────────────────────────
def stage_resume():
    st.subheader("Upload Resume")
    c = st.session_state.candidate

    st.info(f"Candidate: **{c['name']}** · {c['current_role']} · {c['experience']} years exp")

    tab_pdf, tab_text = st.tabs(["📄 Upload PDF", "📝 Paste Text"])

    with tab_pdf:
        uploaded = st.file_uploader("Upload PDF resume", type="pdf", label_visibility="collapsed")
        if uploaded:
            with st.spinner("Parsing PDF and building knowledge base…"):
                full_text, chunks = ingest_pdf(uploaded)
                st.session_state.resume_text = full_text
                st.session_state.resume_chunks = chunks
                st.session_state.vector_db = build_vector_db(tuple(chunks))
            st.success(f"✓ PDF loaded — {len(chunks)} chunks indexed")

    with tab_text:
        pasted = st.text_area("Paste resume content", height=300, placeholder="Copy-paste resume text here…")
        if pasted.strip():
            st.session_state.resume_text = pasted.strip()
            st.success("✓ Resume text saved")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Back"):
            st.session_state.stage = "details"
            st.rerun()
    with col2:
        if st.button("Analyze & Screen →", type="primary", use_container_width=True):
            if not st.session_state.resume_text:
                st.error("Please upload a PDF or paste resume text.")
            else:
                st.session_state.stage = "screening"
                st.rerun()

# ──────────────────────────────────────────────
# STAGE 3 — SCREENING & SCORING
# ──────────────────────────────────────────────
def stage_screening():
    st.subheader("Resume Screening")

    # Run screening if not done yet
    if st.session_state.screening_result is None:
        c = st.session_state.candidate
        with st.spinner("Analyzing resume against job requirements…"):
            prompt = SCREENING_PROMPT.format(
                name=c["name"],
                email=c["email"],
                experience=c["experience"],
                current_role=c["current_role"],
                education=c["education"],
                resume_text=st.session_state.resume_text[:4000],
                job_description=st.session_state.jd[:3000],
                threshold=MATCH_THRESHOLD,
            )
            try:
                result = call_llm_json(prompt)
                st.session_state.screening_result = result
            except Exception as e:
                st.error(f"Screening failed: {e}")
                return

    r = st.session_state.screening_result
    score = r.get("match_score", 0)
    decision = r.get("decision", "Reject")

    # ── Score Header ──
    col_score, col_dec = st.columns([1, 2])
    with col_score:
        color = "#22c55e" if score >= MATCH_THRESHOLD else "#ef4444"
        st.markdown(
            f"""<div style='text-align:center;padding:1rem;border:1px solid {color};border-radius:12px'>
            <div style='font-size:42px;font-weight:700;color:{color}'>{score}%</div>
            <div style='font-size:13px;color:#6b7280'>Match Score</div></div>""",
            unsafe_allow_html=True
        )
    with col_dec:
        if decision == "Proceed":
            st.success("✅ Candidate Shortlisted — Proceeding to Interview")
        else:
            st.error("❌ Candidate Not Shortlisted")
            reason = r.get("reject_reason", "Does not meet minimum requirements.")
            st.warning(f"**Reason:** {reason}")

    # ── Sub-scores ──
    st.markdown("##### Score Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Skills (50%)", f"{r.get('skills_score', 0)}%")
    m2.metric("Experience (20%)", f"{r.get('experience_score', 0)}%")
    m3.metric("Projects (20%)", f"{r.get('projects_score', 0)}%")
    m4.metric("Education (10%)", f"{r.get('education_score', 0)}%")

    st.divider()

    # ── Skills ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**✅ Matching Skills**")
        for s in r.get("matching_skills", []):
            st.markdown(f"`{s}`", unsafe_allow_html=False)
    with col_b:
        st.markdown("**❌ Missing Skills**")
        for s in r.get("missing_skills", []):
            st.markdown(f"`{s}`", unsafe_allow_html=False)

    st.divider()

    # ── Strengths / Weaknesses ──
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**💪 Strengths**")
        for s in r.get("strengths", []):
            st.markdown(f"- {s}")
    with col_d:
        st.markdown("**⚠️ Weaknesses**")
        for w in r.get("weaknesses", []):
            st.markdown(f"- {w}")

    st.divider()

    # ── Extracted Info ──
    with st.expander("📋 Extracted Resume Info"):
        st.markdown(f"**Skills found:** {', '.join(r.get('extracted_skills', []))}")
        st.markdown(f"**Experience:** {r.get('years_experience', 'N/A')}")
        st.markdown(f"**Education:** {r.get('education_summary', 'N/A')}")

    # ── RAG Q&A (from your PDF app) ──
    if st.session_state.vector_db:
        st.divider()
        with st.expander("🔍 Ask a question about this resume (RAG)"):
            query = st.text_input("Ask anything about the candidate's resume:", key="rag_query")
            if query:
                with st.spinner("Searching resume…"):
                    llm = get_llm()
                    retriever = st.session_state.vector_db.as_retriever()
                    template = """Answer the question using only the resume context below.
Context: {context}
Question: {question}
Answer:"""
                    chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | ChatPromptTemplate.from_template(template)
                        | llm
                        | StrOutputParser()
                    )
                    st.write(chain.invoke(query))

    # ── Navigation ──
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Back"):
            st.session_state.screening_result = None
            st.session_state.stage = "resume"
            st.rerun()
    with col2:
        if decision == "Proceed":
            if st.button("Start AI Interview →", type="primary", use_container_width=True):
                st.session_state.stage = "interview"
                st.rerun()

# ──────────────────────────────────────────────
# STAGE 4 — AI INTERVIEW  (your InterviewBot logic)
# ──────────────────────────────────────────────
def stage_interview():
    st.subheader("AI Interview")

    r = st.session_state.screening_result
    c = st.session_state.candidate

    # Generate questions on first entry
    if not st.session_state.questions:
        with st.spinner("Preparing personalized interview questions…"):
            prompt = QUESTIONS_PROMPT.format(
                job_description=st.session_state.jd[:2000],
                candidate_skills=", ".join(r.get("matching_skills", [])),
                candidate_experience=c.get("experience", "N/A"),
                missing_skills=", ".join(r.get("missing_skills", [])),
                num_questions=NUM_QUESTIONS,
            )
            raw = get_llm().invoke(prompt).content
            st.session_state.questions = parse_questions(raw)

    questions = st.session_state.questions
    step = st.session_state.interview_step

    # ── Progress bar ──
    st.progress(step / len(questions), text=f"Question {min(step+1, len(questions))} of {len(questions)}")

    # ── Chat history ──
    st.markdown(f"👋 Hi **{c['name']}**! I'll ask you {len(questions)} questions. Take your time.")
    st.divider()

    # Display past Q&A
    for i in range(step):
        with st.chat_message("assistant"):
            st.write(f"**Q{i+1}.** {questions[i]}")
        if i < len(st.session_state.answers):
            with st.chat_message("user"):
                st.write(st.session_state.answers[i])
        if i < len(st.session_state.answer_feedback):
            fb = st.session_state.answer_feedback[i]
            if isinstance(fb, dict):
                with st.chat_message("assistant", avatar="🤖"):
                    score_icon = score_color(fb.get("score", 5) * 10)
                    st.caption(f"{score_icon} {fb.get('feedback', '')}")

    # Current question
    if step < len(questions):
        with st.chat_message("assistant"):
            st.write(f"**Q{step+1}.** {questions[step]}")

        with st.form(f"answer_form_{step}"):
            answer = st.text_area("Your answer:", height=120, key=f"ans_{step}", placeholder="Type your answer here…")
            submitted = st.form_submit_button("Submit Answer →", type="primary", use_container_width=True)

        if submitted:
            if not answer.strip():
                st.error("Please enter an answer before submitting.")
                return
            # Evaluate answer
            with st.spinner("Evaluating your answer…"):
                eval_prompt = ANSWER_EVAL_PROMPT.format(
                    question=questions[step],
                    answer=answer.strip(),
                    job_context=st.session_state.jd[:500],
                )
                try:
                    feedback = call_llm_json(eval_prompt)
                    # tinyllama sometimes returns a list — normalize to dict
                    if isinstance(feedback, list):
                        feedback = feedback[0] if feedback and isinstance(feedback[0], dict) else {}
                    if not isinstance(feedback, dict):
                        feedback = {}
                except Exception:
                    feedback = {}
                # Ensure required keys always exist
                feedback.setdefault("feedback", "Answer recorded.")
                feedback.setdefault("score", 5)

            st.session_state.answers.append(answer.strip())
            st.session_state.answer_feedback.append(feedback)
            st.session_state.interview_step += 1
            st.rerun()

    else:
        # All questions answered → evaluate
        if st.session_state.final_result is None:
            with st.spinner("Generating final evaluation…"):
                interview_text = "\n".join(
                    f"Q: {q}\nA: {a}"
                    for q, a in zip(questions, st.session_state.answers)
                )
                eval_prompt = EVALUATION_PROMPT.format(
                    job_description=st.session_state.jd[:2000],
                    interview_text=interview_text,
                )
                try:
                    st.session_state.final_result = call_llm_json(eval_prompt)
                except Exception as e:
                    st.error(f"Evaluation error: {e}")
                    return

        st.session_state.stage = "result"
        st.rerun()

# ──────────────────────────────────────────────
# STAGE 5 — FINAL RESULT
# ──────────────────────────────────────────────
def stage_result():
    st.subheader("Final Evaluation")

    c = st.session_state.candidate
    r = st.session_state.final_result or {}
    sr = st.session_state.screening_result or {}

    rec = r.get("recommendation", "Hold")
    rec_map = {"Hire": ("✅ Hire", "success"), "Hold": ("🟡 Hold", "warning"), "Reject": ("❌ Reject", "error")}
    rec_label, rec_type = rec_map.get(rec, ("🟡 Hold", "warning"))

    st.markdown(f"### {c['name']} — {rec_label}")

    if rec_type == "success":
        st.success(rec_label)
    elif rec_type == "warning":
        st.warning(rec_label)
    else:
        st.error(rec_label)

    # ── Combined Score Card ──
    st.markdown("#### Interview Scores")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Technical", f"{r.get('technical_score', 0)}%")
    m2.metric("Problem-solving", f"{r.get('problem_solving_score', 0)}%")
    m3.metric("Communication", f"{r.get('communication_score', 0)}%")
    m4.metric("Overall", f"{r.get('overall_score', 0)}%")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**💪 Strengths Observed**")
        for s in r.get("strengths", []):
            st.markdown(f"- {s}")
    with col_b:
        st.markdown("**📈 Areas to Improve**")
        for w in r.get("weak_areas", []):
            st.markdown(f"- {w}")

    st.divider()
    st.markdown("**📝 Summary**")
    st.info(r.get("summary", "No summary generated."))

    # ── Full Transcript ──
    with st.expander("📋 Full Interview Transcript"):
        for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
            st.markdown(f"**Q{i+1}.** {q}")
            st.markdown(f"> {a}")
            if i < len(st.session_state.answer_feedback):
                fb = st.session_state.answer_feedback[i]
                score_icon = score_color(fb.get("score", 5) * 10)
                st.caption(f"{score_icon} Feedback: {fb.get('feedback', '')}")
            st.divider()

    # ── Screening Summary ──
    with st.expander("📊 Screening Summary"):
        st.markdown(f"**Match Score:** {sr.get('match_score', 0)}%")
        st.markdown(f"**Decision:** {sr.get('decision', 'N/A')}")
        st.markdown(f"**Matching Skills:** {', '.join(sr.get('matching_skills', []))}")
        st.markdown(f"**Missing Skills:** {', '.join(sr.get('missing_skills', []))}")

    st.divider()
    if st.button("🔄 Start New Screening", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="AI Recruitment System",
        page_icon="🤝",
        layout="centered",
    )

    st.title("🤝 AI Recruitment System")
    st.caption("Resume Screening · Skills Matching · AI Interview · Final Evaluation")

    # ── Ollama connectivity check ──
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
    except urllib.error.URLError:
        st.error(f"⚠️ Cannot reach Ollama at `{OLLAMA_BASE_URL}`. Make sure Ollama is running.")
        st.code("ollama serve", language="bash")
        st.info(f"Also ensure the model is pulled:  `ollama pull {OLLAMA_MODEL}`")
        st.stop()

    init_session()
    render_progress()

    stage = st.session_state.stage
    if stage == "details":
        stage_details()
    elif stage == "resume":
        stage_resume()
    elif stage == "screening":
        stage_screening()
    elif stage == "interview":
        stage_interview()
    elif stage == "result":
        stage_result()


if __name__ == "__main__":
    main()