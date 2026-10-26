"""Command line entry point: uv run sec-agent <command> --help"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import get_settings


def cmd_facts(args: argparse.Namespace) -> None:
    from .facts import METRICS, extract_series
    from .factory import build_sec

    sec = build_sec()
    company = sec.lookup(args.company)
    series = extract_series(sec.company_facts(company.cik), args.metric, args.period)
    print(f"{company.name} ({company.ticker}) - {METRICS[args.metric].label}, {args.period}")
    for p in series[-args.last :]:
        print(f"  FY{p.fiscal_year}  ends {p.end}  {p.value:>22,.2f} {p.unit}  ({p.form}, filed {p.filed})")


def cmd_init_db(args: argparse.Namespace) -> None:
    from .db import ChunkStore

    s = get_settings()
    store = ChunkStore(s.database_url, s.embedding_dim)
    if args.reset:
        store.drop()
        print("Dropped the chunks table.")
    store.init_schema()
    print(f"Schema ready ({s.embedding_dim}-dim vectors).")


def cmd_ingest(args: argparse.Namespace) -> None:
    from .db import ChunkStore
    from .factory import build_embedder, build_sec
    from .ingest import ingest_company

    s = get_settings()
    store = ChunkStore(s.database_url, s.embedding_dim)
    store.init_schema()
    embedder = build_embedder(s)
    sec = build_sec()
    for company in args.companies:
        n = ingest_company(company, sec, store, embedder, tuple(args.forms), args.limit, args.force)
        print(f"{company}: {n} chunks stored")


def cmd_stats(args: argparse.Namespace) -> None:
    from .db import ChunkStore

    s = get_settings()
    for ticker, form, year, section, count in ChunkStore(s.database_url, s.embedding_dim).stats():
        print(f"{ticker:6} {form:5} FY{year}  {section:13} {count:5} chunks")


def cmd_search(args: argparse.Namespace) -> None:
    from .factory import build_retriever, build_sec
    from .models import Filters

    ticker = build_sec().lookup(args.company).ticker if args.company else None
    hits = build_retriever(mode=args.mode).search(
        args.query, Filters(ticker=ticker, fiscal_year=args.year, section=args.section), k=args.k
    )
    for h in hits:
        c = h.chunk
        print(f"[{h.source_id}] {h.score:.3f}  {c.ticker} {c.form} FY{c.fiscal_year} {c.section}")
        print("   " + c.content[:300].replace("\n", " ") + "...\n")


def cmd_ask(args: argparse.Namespace) -> None:
    from .factory import build_agent

    agent = build_agent(mode=args.mode)
    if args.graph:
        from .graph_agent import ask_with_graph

        result = ask_with_graph(agent, args.question)
    else:
        result = agent.ask(args.question)
    if args.json:
        print(result.model_dump_json(indent=2))
        return
    print(result.answer)
    for c in result.citations:
        where = c.section or "XBRL"
        print(f"  [{c.source_id}] {c.company} {c.form} filed {c.filing_date} ({where}) {c.url}")
    if result.invalid_citations:
        print(f"  cited ids that were never retrieved: {result.invalid_citations}")
    if result.tool_calls:
        calls = ", ".join(f"{c['tool']}{'' if c['ok'] else ' (error)'}" for c in result.tool_calls)
        print(f"\ntools: {calls}")
    print(
        f"\n{result.model} ({result.route_reason}) | {result.steps} steps | "
        f"${result.cost_usd:.4f} | {result.latency_ms} ms"
    )


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("sec_research_agent.api:app", host=args.host, port=args.port, reload=args.reload)


def cmd_mcp(args: argparse.Namespace) -> None:
    from .mcp_server import main

    main()


def cmd_eval(args: argparse.Namespace) -> None:
    from .evals import run_evals

    sys.exit(run_evals(args))


def cmd_candidates(args: argparse.Namespace) -> None:
    from .evals import make_candidates

    make_candidates(args)


def build_parser() -> argparse.ArgumentParser:
    from .facts import METRICS
    from .retrieval import MODES

    p = argparse.ArgumentParser(prog="sec-agent", description="SEC filings research agent")
    p.add_argument("-v", "--verbose", action="store_true", help="show progress logs")
    # -v also works after the command: sec-agent ingest AAPL -v
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("facts", parents=[common], help="print a metric from XBRL company facts")
    s.add_argument("company")
    s.add_argument("metric", choices=list(METRICS))
    s.add_argument("--period", choices=["annual", "quarterly"], default="annual")
    s.add_argument("--last", type=int, default=5)
    s.set_defaults(func=cmd_facts)

    s = sub.add_parser("init-db", parents=[common], help="create the chunks table")
    s.add_argument("--reset", action="store_true", help="drop all stored chunks first")
    s.set_defaults(func=cmd_init_db)

    s = sub.add_parser("ingest", parents=[common], help="download, chunk, embed and store filings")
    s.add_argument("companies", nargs="+")
    s.add_argument("--forms", nargs="+", default=["10-K"])
    s.add_argument("--limit", type=int, default=3, help="filings per company")
    s.add_argument("--force", action="store_true", help="re-ingest stored filings")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("stats", parents=[common], help="what is stored")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("search", parents=[common], help="search filing passages")
    s.add_argument("query")
    s.add_argument("--company")
    s.add_argument("--year", type=int)
    s.add_argument("--section", choices=["business", "risk_factors", "mdna"])
    s.add_argument("--mode", choices=MODES)
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("ask", parents=[common], help="answer a question")
    s.add_argument("question")
    s.add_argument("--mode", choices=MODES)
    s.add_argument("--graph", action="store_true", help="use the LangGraph version")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("serve", parents=[common], help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("mcp", parents=[common], help="run the MCP server over stdio")
    s.set_defaults(func=cmd_mcp)

    s = sub.add_parser("eval", parents=[common], help="run the golden-set evals")
    s.add_argument("--golden", default="evals/golden.jsonl")
    s.add_argument("--mode", choices=MODES)
    s.add_argument("--include-unverified", action="store_true")
    s.add_argument("--limit", type=int)
    s.add_argument("--min-accuracy", type=float, default=0.0)
    s.add_argument("--out", default="evals/results")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("candidates", parents=[common], help="draft numeric golden questions from XBRL (verify by hand)")
    s.add_argument("companies", nargs="+")
    s.add_argument("--metrics", nargs="+", default=["revenue", "net_income", "total_assets"])
    s.add_argument("--years", type=int, default=2)
    s.add_argument("--out", default="evals/candidates.jsonl")
    s.set_defaults(func=cmd_candidates)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    get_settings()  # loads .env into the environment
    from .tracing import setup_tracing

    setup_tracing()
    args.func(args)


if __name__ == "__main__":
    main()
