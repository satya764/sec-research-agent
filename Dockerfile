FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra mcp

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "sec_research_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
