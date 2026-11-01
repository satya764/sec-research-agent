# SEC Filings Research Agent

Ask a question about a public company. Get an answer built from its 10-K and 10-Q filings, with every number and claim cited.

> Status: working end to end on Apple, Microsoft and NVIDIA 10-Ks, runs free on a laptop. Oct 2026 – Jan 2027.

![Demo: asking for Apple revenue growth, answer with XBRL sources, cost and latency](docs/demo.gif)

## What it does

- Parses a question into structured fields: companies, metric, fiscal years, question type
- Searches the narrative sections of filings (Item 1 Business, 1A Risk Factors, 7 MD&A) with hybrid search, plus optional Cohere reranking
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

Golden set: 37 hand-verified questions over the latest Apple, Microsoft and NVIDIA 10-Ks: 21 single lookups,
10 hard (margins, multi-year growth, cross-company comparisons), 3 narrative, 3 unanswerable.
Every expected number was checked against the filing's financial statements.

| Metric | Qwen3 8B, local | Claude Haiku 5.5 | Claude Sonnet 5.5 |
| --- | --- | --- | --- |
| Answer accuracy | 97% (36/37) | **100%** (37/37) | **100%** (37/37) |
| Citation correctness | 97% | 97% | 95% |
| Refusal accuracy | 97%* | 100% | 100% |
| Retrieval recall@5 | 100% | 100% | 100% |
| Cost per question | $0 | **$0.0009** | $0.0163 |
| Cost for all 37 | $0 | $0.03 | $0.60 |
| p95 latency | 171 s | **4.9 s** | 9.4 s |

Hybrid search, no reranker, local bge-small embeddings. Oct 31 2026. Claude Haiku judges both Claude runs; the local
model judges itself, so the local run stays free and offline. Narrative answers are graded against reference answers
written from the 10-K text. Raw records for every run: [`evals/reports/`](evals/reports/).

\* Qwen3 hit the 8-step limit on Microsoft's segments in the full run and answered "could not finish", which counts as
a wrong refusal. It answered correctly on a rerun. Its one wrong answer: Apple's supplier risks, where it left out
custom components and supply shortages.

**Decision: Haiku for everything.** On this set Sonnet costs 18x more and is twice as slow with no accuracy gain,
so the recommended config sets both `CHEAP_MODEL` and `STRONG_MODEL` to Haiku. The router stays in place for
question types where a future eval shows Haiku falling behind.

**Reading the citation scores.** Haiku's one miss is real: for Microsoft's segments it named all three, but the
cited passages only contained two, so it filled in the third from memory. Sonnet's two misses are judge errors:
both answers were correct arithmetic on cited numbers, which the judge is told to accept. The local judge made the
same mistake once, on a net income difference.

**Local vs Claude.** The free local model is nearly as accurate (36/37) but takes minutes per question instead of
seconds. Good for development and private data; Haiku for anything a user waits on.

**Local model history**

| Run | Questions | Answer accuracy | Change |
| --- | --- | --- | --- |
| 1 | 27 base | 92% (22/24) | baseline |
| 2 | 27 base | 100% (24/24) | clarified the `value` field in `final_answer` |
| 3 | 10 hard | 80% (8/10) | first hard set |
| 4 | 3 failed | all pass | metric hint, newest-first values, "don't guess the year" |

## Load test

`scripts/loadtest.py`, 500 requests at concurrency 20 against one API process on a MacBook, Postgres state backend.
Half are `/facts` lookups, half are `/ask` on a cached answer, so this measures the API, database and cache,
not model speed (model latency is in the table above).

| | Result |
| --- | --- |
| Throughput | 124.7 req/s |
| Latency | p50 181 ms, p95 328 ms, p99 515 ms |
| Errors | 0 |
| Rate limiter (20 `/ask` per minute per client) | rejected 231 of the over-limit calls with 429 |

The API holds no state of its own (rate limits, cache and chat history are in Postgres), so throughput scales by
adding instances behind a load balancer. Uncached answers are bound by the model: about 5 s each on Haiku.

## Stack

Python 3.13 · FastAPI · Postgres + pgvector · Claude Haiku/Sonnet or local Qwen3 via Ollama · local bge-small embeddings (fastembed) · Cohere rerank · LangGraph · MCP · Arize Phoenix · Docker · Streamlit · GitHub Actions

## Run it

You need: [uv](https://docs.astral.sh/uv/) and Docker. Everything can run free on your machine: embeddings with fastembed, answers with [Ollama](https://ollama.com). Claude (more accurate), Cohere and OpenAI are optional, set in `.env`.

Free local model (once):

```bash
ollama pull qwen3:8b
printf 'FROM qwen3:8b\nPARAMETER num_ctx 16384\n' > Modelfile   # default 4K context is too small for the agent
ollama create qwen3-16k -f Modelfile && rm Modelfile
```

```bash
cp .env.example .env          # set SEC_USER_AGENT and pick a model block (local or Claude)
uv sync --all-extras          # installs everything, updates uv.lock
uv run pytest                 # unit tests, no keys needed

# 1. Numbers only (no database needed)
uv run sec-agent facts AAPL revenue

# 2. Database + filings
docker compose up -d db
uv run sec-agent init-db
uv run sec-agent ingest AAPL MSFT NVDA --limit 1 -v
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
uv run sec-agent eval                                # all verified rows
uv run sec-agent eval --limit 5                      # quick check
uv run sec-agent eval --ids msft-revenue-growth-2026 # rerun specific rows
uv run sec-agent eval --mode vector                  # compare retrieval modes: vector, hybrid, hybrid_rerank
```

Only verified rows count. Results go to `evals/results/` (gitignored); the runs behind the Results table are in `evals/reports/`. The Evals workflow runs on pull requests
when an `ANTHROPIC_API_KEY` repo secret is set, and fails below 80% accuracy.

## Project layout

```
src/sec_research_agent/
  sec_client.py   EDGAR: ticker lookup, company facts, filing lists, documents (rate limited, cached)
  facts.py        clean annual/quarterly series out of XBRL facts
  filings.py      HTML to text, find Items 1, 1A, 7 (skips the table of contents)
  chunking.py     paragraph -> sentence -> hard-cut chunking with overlap
  embeddings.py   local embeddings (fastembed), OpenAI option, hash embedder for tests
  db.py           pgvector store: vector search, full-text search, metadata filters
  retrieval.py    vector / keyword / hybrid (RRF) / hybrid + Cohere rerank
  llm.py          Claude and OpenAI-compatible (Ollama) clients: cost per call, fallback, structured output
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
evals/            golden set, question candidates, final eval reports
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
| Local embeddings (bge-small, 384 dims) by default | Free, private, no rate limits; OpenAI stays one setting away for comparison | OpenAI text-embedding-3-small |
| Postgres for vectors and keywords | One database, metadata filters in SQL | Pinecone, Chroma, Elasticsearch |
| `final_answer` as a tool with citation ids | Citations are structured and checkable; unknown ids are flagged | Free-text answers with inline citations |
| `tool_choice: any` | The model must act through tools every turn, so it can't answer from memory | Let the model reply in text |
| Haiku for everything, router kept | Haiku matched Sonnet on all 37 questions at 1/18 the cost | Sonnet for multi-step questions |
| Model provider is a setting (Claude or any OpenAI-compatible API) | Runs free on a laptop with Ollama; the same evals compare local vs Claude | Claude only |
| Plain loop first, LangGraph second | Shows what the framework adds | Start with a framework |

## What didn't work

- Ollama's default 4K context made the agent loop until the step limit. Fixed with a 16K-context model (`num_ctx 16384`).
- Page furniture leaked into chunks: Apple's footers ("Apple Inc. | 2025 Form 10-K | 11"), NVIDIA's "Table of Contents" links, page numbers, Microsoft's running "PART I / Item 1A" headers. Fixed with a line filter after section extraction (0 left across all three 10-Ks).
- The local model sometimes replies in text instead of calling a tool; the agent nudges it and continues.
- Citation check flagged a correct answer (Microsoft segments): the judge only saw the first 1,000 characters of each passage. Now it sees the full passage.

- Hard questions, run 1: 80% (8/10). With no year given, the facts tool listed 5 years oldest-first and the
  model answered "largest total assets" with Microsoft's FY2023 value. Fixed: newest first, plus `latest_fiscal_year`.
  It also looped asking for a "gross margin" metric that doesn't exist. Fixed: the error now says which metrics to
  fetch and to divide with the calculator. The local judge also marked a correct subtraction as unsupported.
- The real cause of the "largest total assets" miss: asked for the *latest* year, the local model passed
  `fiscal_year: 2023` itself, likely the newest year in its training data, despite today's date in the prompt.
  Fixed: the tool tells the model to omit the year for "latest", and adds a note when an older year is requested.
  After the fix the local model got it right, but wrote Apple's assets as "$359,241 billion" (the number is in
  millions, so 1,000x off). `value` was correct, so accuracy passed; the citation judge caught the text.
  Claude Haiku answered the same question correctly ("$359.2 billion") in 3.7 s for $0.0009.

- Claude Sonnet 5.5 rejects forced tool use (`tool_choice` any/tool returns a 400). The client now detects that,
  remembers the model, and switches to `auto` with the rule stated in the prompt. Fallback to a second model now
  happens only on rate limits and outages, not on bad requests.

## License

MIT
