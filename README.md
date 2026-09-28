# XCRS — Explainable Course Recommendation System

XCRS recommends **career roles** and **online courses** based on what you have already learned and what you are curious about. It also explains, in plain language, **why** it recommends each role and course.

> **Status:** the `modernization` branch is a work in progress. The goal is to make XCRS production-ready (cloud deployment, a proper data layer, CI/CD, and a new UI). The system described below is the current, research-prototype version.

---

## How it works

You fill in four free-text fields, each a comma-separated list of courses, subjects or concepts:

| Field             | Meaning                                | Weight  |
| ----------------- | -------------------------------------- | ------- |
| Curious about     | Things you want to learn               | `1.0`   |
| Took and liked    | Things you studied and enjoyed         | `0.75`  |
| Took, neutral     | Things you studied, no strong feelings | `0.5`   |
| Took and disliked | Things you studied and didn't enjoy    | `-0.25` |

XCRS then:

1. **Embeds** every item you entered with an LLM embedding model.
2. **Matches** your items to concepts from [roadmap.sh](https://roadmap.sh) career roadmaps, using cosine similarity with a statistical threshold (mean + 2.5σ of the course × concept similarity distribution).
3. **Scores the 10 career roles** by weighted concept coverage, squashed through a sigmoid, and picks the top 3.
4. **Recommends 3 Udemy courses per role.** It targets the roadmap concepts you haven't covered yet and down-weights courses similar to ones you disliked.
5. **Explains** each role and course recommendation with an LLM (`gpt-4o`).

### Supported career roles

AI Data Scientist · Android Developer · Backend Developer · Blockchain Developer · DevOps Engineer · Frontend Developer · Full Stack Developer · Game Developer · QA Engineer · UX Designer

### Embedding models compared

The research prototype runs the same pipeline with five embedding providers so they can be compared side by side. The UI labels them _Model-1 … Model-5_.

| #   | Provider  | Model                    |
| --- | --------- | ------------------------ |
| 1   | Google    | `text-embedding-004`     |
| 2   | Voyage AI | `voyage-large-2`         |
| 3   | OpenAI    | `text-embedding-3-large` |
| 4   | Mistral   | `mistral-embed`          |
| 5   | Cohere    | `embed-english-v3.0`     |

---

## Architecture

```
                         ┌─────────────────────────────┐
  Browser ──► Nuxt 3 ──► │ /api/recommend (Nuxt server)│
                         └──────────────┬──────────────┘
                                        │ calls each model endpoint in turn
                                        ▼
                         ┌─────────────────────────────┐
                         │ FastAPI backend             │
                         │  POST /recommendations/{m}  │  m ∈ google|voyage|openai|mistral|cohere|mock
                         │  POST /save_inputs          │
                         └──────────────┬──────────────┘
                                        │ loads at startup
                                        ▼
                         ┌─────────────────────────────┐
                         │ Pre-computed embeddings     │  (CSV files, generated offline)
                         │ courses × roadmap concepts  │
                         └─────────────────────────────┘
```

| Component                                       | Stack                     | Purpose                                                                                    |
| ----------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------ |
| [`embedding-generation/`](embedding-generation) | Python, pandas            | Offline: cleans course data, flattens roadmaps, embeds everything with all 5 providers     |
| [`backend/`](backend)                           | Python, FastAPI           | Online: embeds user input, matches it, scores roles, picks courses, generates explanations |
| [`frontend/`](frontend)                         | Nuxt 3, Nuxt UI, Tailwind | Input form and results view with a model switcher                                          |

### Data

| Dataset                                                                   | Rows                              | Source                                                           |
| ------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------- |
| Udemy courses (raw)                                                       | 973                               | Udemy                                                            |
| Udemy courses (filtered: Development, IT & Software, Office Productivity) | 453                               | derived                                                          |
| Roadmap nodes                                                             | 1,104 (869 concepts + 235 topics) | [roadmap.sh](https://github.com/kamranahmedse/developer-roadmap) |

---

## New backend (in progress, `modernization` branch)

The modernized data layer runs locally with PostgreSQL + pgvector in Docker and a self-hosted embedding model in [Ollama](https://ollama.com). Design decisions are recorded in [`docs/adr/`](docs/adr/README.md); the schema is in [`docs/schema.md`](docs/schema.md).

```bash
brew install ollama uv
ollama pull qwen3-embedding:0.6b          # Ollama runs natively (Apple GPU)
cp .env.example .env
docker compose up -d postgres

cd backend
uv sync
uv run alembic upgrade head               # create the schema
uv run xcrs import-prototype              # CSV data → Postgres
uv run xcrs register-model qwen3-embedding:0.6b --id 1 --status active
uv run xcrs embed-catalog qwen3-embedding:0.6b
uv run xcrs search "Docker"               # smoke test
uv run pytest
```

## Running the research prototype

### Prerequisites

- Python 3.10+
- Node.js 18+ and `pnpm`
- API keys for Google, Voyage, OpenAI, Mistral and Cohere

Put each key in its own file (the folder is gitignored):

```
embedding-generation/api_keys/google_api_key.txt
embedding-generation/api_keys/voyage_api_key.txt
embedding-generation/api_keys/openai_api_key.txt
embedding-generation/api_keys/mistral_api_key.txt
embedding-generation/api_keys/cohere_api_key.txt
```

### 1. Generate embeddings (one-time)

```bash
pip install pandas numpy scikit-learn tiktoken google-generativeai voyageai openai "mistralai<1" cohere
cd embedding-generation/src
python main.py
```

This creates `embedding-generation/data/<provider>_emb/` folders with the course and roadmap-node embeddings.

### 2. Start the backend

```bash
pip install fastapi uvicorn pandas numpy scikit-learn google-generativeai voyageai openai "mistralai<1" cohere
mkdir -p backend/log backend/user_inputs
cd backend/src          # paths are relative, so it must run from here
python main.py          # http://localhost:8000, docs at /docs
```

> Use `python main.py`. Running `uvicorn main:app` directly skips `main()`, so the embeddings never load.

### 3. Start the frontend

```bash
cd frontend
pnpm install
pnpm run dev            # http://localhost:3000
```

---

## Roadmap

Modernization work happens on the `modernization` branch. Candidate directions include containerization, CI/CD, cloud infrastructure as code, a database with vector search in place of CSV files, a redesigned UI, an improved LLM layer, and eventually a better course scraper and AI-assisted roadmap generation. The final choices are still open.

---

## Research origin & citation

XCRS started as an academic research project. The original replication package (code, data, the user-study protocol, questions and responses) is archived on Zenodo:

[![DOI](https://zenodo.org/badge/783277216.svg)](https://doi.org/10.5281/zenodo.14291086)

## Credits

- Courses: [Udemy](https://udemy.com)
- Roadmaps: [roadmap.sh](https://roadmap.sh) ([GitHub](https://github.com/kamranahmedse/developer-roadmap))

## License

[GPL-3.0-or-later](LICENSE.md)
