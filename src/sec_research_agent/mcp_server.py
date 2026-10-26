"""Expose the agent's tools over MCP, so Claude Desktop or any MCP client can use them.

Run: uv run sec-agent mcp
"""

from __future__ import annotations

import json

from .factory import build_agent, build_retriever, build_sec
from .tools import ToolContext

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

server = Server("sec-research-agent")
_ctx: ToolContext | None = None


def _context() -> ToolContext:
    global _ctx
    if _ctx is None:
        _ctx = ToolContext(sec=build_sec(), retriever=build_retriever())
    return _ctx


@server.tool()
def search_filings(
    query: str,
    company: str | None = None,
    fiscal_year: int | None = None,
    section: str | None = None,
) -> str:
    """Search 10-K/10-Q narrative text (business, risk_factors, mdna). Returns cited passages."""
    args = {"query": query, "company": company, "fiscal_year": fiscal_year, "section": section}
    return _context().run("search_filings", {k: v for k, v in args.items() if v is not None})


@server.tool()
def get_financial_data(company: str, metric: str, fiscal_year: int | None = None) -> str:
    """Reported XBRL values (base units) for a metric such as revenue or net_income."""
    args = {"company": company, "metric": metric, "fiscal_year": fiscal_year}
    return _context().run("get_financial_data", {k: v for k, v in args.items() if v is not None})


@server.tool()
def calculator(expression: str) -> str:
    """Evaluate plain arithmetic, e.g. (120 - 100) / 100 * 100."""
    return _context().run("calculator", {"expression": expression})


@server.tool()
def ask(question: str) -> str:
    """Answer a question end to end with citations, cost and latency."""
    result = build_agent().ask(question)
    return json.dumps(result.model_dump(exclude={"tool_calls"}), default=str)


def main() -> None:
    server.run()
