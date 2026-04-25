AI INTERVIEW AGENT

An end-to-end AI-powered recruitment pipeline that screens candidates from their resume and conducts a full AI-driven interview — all running **100% locally** on your machine using Ollama.


Pipeline Overview

┌─────────────────────────────────────────────────────────────┐
│                    AI Recruitment System                     │
├──────────┬──────────┬──────────────┬───────────┬───────────┤
│ Stage 1  │ Stage 2  │   Stage 3    │  Stage 4  │  Stage 5  │
│ Details  │ Resume   │  Screening   │ Interview │  Result   │
│          │ Upload   │  & Scoring   │   (AI)    │           │
├──────────┼──────────┼──────────────┼───────────┼───────────┤
│ Name     │ PDF via  │ Skills match │ 5 custom  │ Hire /    │
│ Email    │ RAG  or  │ Experience   │ questions │ Hold /    │
│ Exp      │ paste    │ Score ≥ 70%  │ Per-ans   │ Reject    │
│ Role     │ text     │ → Proceed    │ feedback  │ + full    │
│ JD       │          │ < 70% Reject │           │ report    │
└──────────┴──────────┴──────────────┴───────────┴───────────┘
```

 Features

- **PDF Resume Ingestion** — Upload a PDF and it's chunked, embedded, and stored in a local Chroma vector DB using `all-MiniLM-L6-v2` embeddings
- **RAG Q&A** — Ask any question about the candidate's resume directly from the screening page
- **AI Screening** — Weighted match scoring across Skills (50%), Experience (20%), Projects (20%), Education (10%)
- **Smart Decision** — Score ≥ 70% proceeds to interview; below that gets rejected with a reason
- **Personalized Interview** — 5 questions generated based on the candidate's actual skills and gaps
- **Per-answer Feedback** — Real-time evaluation after each answer
- **Final Report** — Technical, Problem-solving, Communication scores + Hire/Hold/Reject recommendation + full transcript


Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM | Ollama (local, free) |
| PDF loading | LangChain `PyPDFLoader` |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Vector store | Chroma (local) |
| RAG chain | LangChain LCEL |
| Env management | `python-dotenv` |

 System Requirements

| Resource | Minimum |
|---|---|
| RAM | 2 GB free |
| Python | 3.10 or higher |
| Ollama | Latest version |
| OS | Windows / macOS / Linux |

 Quickstart

### Step 1 — Install Ollama

Download from [https://ollama.com](https://ollama.com) and install it.

### Step 2 — Pull a model

Choose based on your available RAM:

| Model | RAM needed | Command | Recommended? |
|---|---|---|---|
| `tinyllama` | ~0.9 GB | `ollama pull tinyllama` | ✅ Low RAM machines |
| `phi` | ~2.0 GB | `ollama pull phi` | ✅ Better quality |
| `llama3.2` | ~2.3 GB | `ollama pull llama3.2` | ✅ Best quality |

```bash
ollama pull tinyllama
```

### Step 3 — Clone & install dependencies

```bash
git clone <your-repo-url>
cd ai_recruitment_system

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 4 — Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=tinyllama
```

### Step 5 — Run

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

 Project Structure

ai_recruitment_system/
├── app.py                      ← Main Streamlit app (all 5 stages)
├── requirements.txt            ← Python dependencies
├── .env.example                ← Environment variables template
├── .env                        ← Your local config (gitignored)
├── README.md                   ← This file
└── .streamlit/
    └── config.toml             ← Suppresses torchvision watcher warnings
```
 Configuration

Edit these constants at the top of `app.py`:

| Constant | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `tinyllama` | Model to use |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model (CPU, free) |
| `MATCH_THRESHOLD` | `70` | Minimum % score to pass screening |
| `NUM_QUESTIONS` | `5` | Number of interview questions |

 How It Works — Stage by Stage

### Stage 1 · Candidate Details
Fill in the candidate's name, email, years of experience, current role, education, and paste the full job description.

### Stage 2 · Resume Upload
Upload a PDF resume (parsed via `PyPDFLoader`, chunked and stored in Chroma for RAG) or paste resume text directly.

### Stage 3 · AI Screening
The LLM analyzes the resume against the JD and returns:
- Extracted skills, experience, education
- Weighted match score (skills 50% · experience 20% · projects 20% · education 10%)
- Matching skills ✅ and missing skills ❌
- Strengths and weaknesses
- **Decision: Proceed (≥70%) or Reject (<70%)**

You can also ask free-form questions about the resume using the built-in RAG panel.

### Stage 4 · AI Interview *(only if passed screening)*
- 5 personalized questions generated from candidate's skills and gaps
- One question at a time with a text input
- Real-time per-answer feedback and score after each submission

### Stage 5 · Final Result
- Technical / Problem-solving / Communication scores
- Overall interview score
- Strengths and areas to improve
- **Recommendation: Hire / Hold / Reject**
- Full transcript with per-answer feedback

 Common Issues

### `model requires more system memory than is available`
Your RAM is too low for the selected model. Switch to a smaller one:
```env
OLLAMA_MODEL=tinyllama
```

### `ModuleNotFoundError: No module named 'torchvision'`
These are harmless Streamlit watcher warnings — not errors. The `.streamlit/config.toml` file suppresses them. Your app still works fine.

### `Screening failed: JSONDecodeError`
The model returned malformed JSON. The app has a 3-layer fallback parser, but very small models can still fail. Try switching to `phi` if you have enough RAM.

### Ollama not reachable
Make sure Ollama is running:
```bash
ollama serve
```

 Project Origins

This project merges three separate applications:

| Original Project | What it contributed |
|---|---|
| **MedQuery AI** (PDF Q&A app) | `PyPDFLoader`, Chroma vector DB, RAG chain, HuggingFace embeddings |
| **InterviewBot** (AI interview chatbot) | Sequential Q&A session state, question generation, candidate evaluation prompts |
| **Recruitment Pipeline** (screening system) | Weighted match scoring, skills gap analysis, 5-stage UI, final Hire/Hold/Reject report |

---

## 📄 License

MIT License — free to use, modify, and distribute.
