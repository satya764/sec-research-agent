# SEC Filings Research Agent

Ask a question about a public company. Get an answer built from its 10-K and 10-Q filings, with every number and claim cited.

> Status: code complete, being tested on real filings. Oct 2026 – Jan 2027.

## What it does

- Parses a question into structured fields: companies, metric, fiscal years, question type
- Searches the narrative sections of filings (Item 1 Business, 1A Risk Factors, 7 MD&A) with hybrid search + reranking
- Pulls numbers from the SEC EDGAR Company Facts API (XBRL), never from model memory
- Does math with a calculator tool, not in the model's head
- Cites the filing and section for every claim, and flags citations it never retrieved
- Answers "Not found in filings." instead of guessing
- Routes simple questions to a cheap model, hard ones to a strong model, with a fallback model on API errors
- Redacts personal data and blocks prompt injection before anything reaches a model
- Exposes its tools as an MCP server
- Same agent in two versions: a plain tool-use loop and a LangGraph state machine

## How it works

```mermaid
flowchart LR
    Q[Question] --> G[Guardrails<br/>PII, injection]
    G --> P[Parse question<br/>cheap model]
    P --> R{Router}
    R --> A[Agent loop]
    A -->|narrative| S[Hybrid search + rerank<br/>pgvector + Postgres full-text]
    A -->|numbers| X[EDGAR Company Facts<br/>XBRL]
    A -->|math| C[Calculator]
    S --> A
    X --> A
    C --> A
    A --> ANS[Answer + citations<br/>cost, latency]
    A -->|nothing found| NF[Not found in filings]
```

## Results

Measured on a hand-verified golden set (`evals/golden.jsonl`). Filled in as evals run.

| Metric | vector only | hybrid | hybrid + rerank |
| --- | --- | --- | --- |
| Retrieval recall@5 | – | – | – |
| Retrieval MRR | – | – | – |
| Answer accuracy | – | – | – |
| Citation correctness | – | – | – |
| Refusal accuracy | – | – | – |
| Cost per question | – | – | – |
| p95 latency | – | – | – |

## Stack

Python 3.13 · FastAPI · Postgres + pgvector · Claude Haiku/Sonnet · OpenAI embeddings · Cohere rerank · LangGraph · MCP · Arize Phoenix · Docker · Streamlit · GitHub Actions

## Run it

You need: [uv](https://docs.astral.sh/uv/), Docker, an Anthropic API key, an OpenAI API key (embeddings). Cohere is optional.

```bash
cp .env.example .env          # fill in keys and SEC_USER_AGENT
uv sync --all-extras          # installs everything, updates uv.lock
uv run pytest                 # unit tests, no keys needed

# 1. Numbers only (no database needed)
uv run sec-agent facts AAPL revenue

# 2. Database + filings
docker compose up -d db
uv run sec-agent ingest AAPL MSFT NVDA --limit 2 -v
uv run sec-agent stats
uv run sec-agent search "supplier concentration risk" --company AAPL

# 3. Ask
uv run sec-agent ask "How much did Apple's revenue change from fiscal 2023 to 2024?"
uv run sec-agent ask "What are Microsoft's main risk factors around AI?" --graph

# 4. API + UI
uv run sec-agent serve                 # http://localhost:8000/docs
uv run streamlit run app/ui.py         # http://localhost:8501

# 5. MCP server (for Claude Desktop or any MCP client)
uv run sec-agent mcp
```

Or run the API and database together: `docker compose up --build`.

## Evals

```bash
uv run sec-agent candidates AAPL MSFT NVDA          # drafts numeric questions from XBRL
# open evals/candidates.jsonl, check each number against the filing,
# set "verified": true, and copy the rows into evals/golden.jsonl
uv run sec-agent eval --mode vector
uv run sec-agent eval --mode hybrid
uv run sec-agent eval --mode hybrid_rerank
```

Only verified rows count. Results go to `evals/results/`. The Evals workflow runs on pull requests and fails below 80% accuracy.

## Project layout

```
src/sec_research_agent/
  sec_client.py   EDGAR: ticker lookup, company facts, filing lists, documents (rate limited, cached)
  facts.py        clean annual/quarterly series out of XBRL facts
  filings.py      HTML to text, find Items 1, 1A, 7 (skips the table of contents)
  chunking.py     paragraph -> sentence -> hard-cut chunking with overlap
  embeddings.py   OpenAI embeddings, plus a hash embedder for tests
  db.py           pgvector store: vector search, full-text search, metadata filters
  retrieval.py    vector / keyword / hybrid (RRF) / hybrid + Cohere rerank
  llm.py          Anthropic wrapper: cost per call, fallback model, structured output
  router.py       cheap vs strong model
  guardrails.py   PII redaction, prompt injection checks
  tools.py        search_filings, get_financial_data, calculator, final_answer
  agent.py        plain tool-use loop
  graph_agent.py  same agent as a LangGraph graph
  api.py          FastAPI: /ask, /facts, rate limit, answer cache, session memory
  mcp_server.py   MCP tools
  evals.py        golden-set evals and candidate generation
  tracing.py      Phoenix tracing (optional)
app/ui.py         Streamlit chat UI
evals/            golden set and results
docs/deploy.md    AWS deployment
```

## Decisions

| Decision | Why | Alternatives considered |
| --- | --- | --- |
| Numbers from XBRL, never from text chunks | Financial tables get split across chunks; XBRL values are exact and tagged | Parse HTML tables |
| Drop numeric tables during ingestion | Keeps chunks narrative; numbers have one source | Keep tables as text |
| Only Items 1, 1A, 7 (10-K) and MD&A / risk factors (10-Q) | These answer descriptive questions; the rest is boilerplate or numbers | Whole filing |
| Longest span per item heading | Every item also appears in the table of contents | First match, regex on TOC page numbers |
| Hybrid search with reciprocal rank fusion | Vector search misses exact terms (product names, "Item 1A"); keyword misses paraphrases | Vector only |
| Postgres for vectors and keywords | One database, metadata filters in SQL | Pinecone, Chroma, Elasticsearch |
| `final_answer` as a tool with citation ids | Citations are structured and checkable; unknown ids are flagged | Free-text answers with inline citations |
| `tool_choice: any` | The model must act through tools every turn, so it can't answer from memory | Let the model reply in text |
| Haiku for lookups, Sonnet for multi-step | Most questions are single lookups | One model for everything |
| Plain loop first, LangGraph second | Shows what the framework adds | Start with a framework |

## What didn't work

Logged as it happens.

## License

MIT
