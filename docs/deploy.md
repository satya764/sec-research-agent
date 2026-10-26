# Deploying to AWS

The API runs as one container (App Runner). The database is Postgres with pgvector (RDS).

## 1. Database

Create an RDS PostgreSQL 17 instance (db.t4g.micro is enough to start). RDS supports the
`vector` extension; the app runs `CREATE EXTENSION IF NOT EXISTS vector` on first connect.

Load filings from your laptop into it:

```bash
DATABASE_URL=postgresql://USER:PASS@HOST:5432/sec_agent uv run sec-agent ingest AAPL MSFT NVDA --limit 2
```

## 2. Image

```bash
aws ecr create-repository --repository-name sec-research-agent
aws ecr get-login-password | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker build --platform linux/amd64 -t sec-research-agent .
docker tag sec-research-agent ACCOUNT.dkr.ecr.REGION.amazonaws.com/sec-research-agent:latest
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/sec-research-agent:latest
```

Run `uv sync` locally first so `uv.lock` matches `pyproject.toml` (the build uses `--frozen`).

## 3. Service

App Runner -> Create service -> Container registry (ECR) -> the image above.

- Port: 8000
- Health check: HTTP `/health`
- Environment: `ANTHROPIC_API_KEY`, `SEC_USER_AGENT`, `DATABASE_URL`, optional `COHERE_API_KEY`
  (the cloud demo uses Claude; a local Ollama model would need a GPU server)
  (store keys in Secrets Manager and reference them)
- Networking: give App Runner a VPC connector if RDS is private

## 4. UI

Deploy `app/ui.py` on Streamlit Community Cloud with `API_URL` set to the App Runner URL.

## Costs to watch

App Runner and RDS bill while running. Pause the App Runner service and stop RDS when not demoing.
