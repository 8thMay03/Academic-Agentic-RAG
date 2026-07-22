# Deployment Notes

## Docker

The root `docker-compose.yml` builds both services:

```bash
docker compose up --build
```

Runtime data is mounted at `backend/data:/app/data`. The backend image code is not bind-mounted over `/app`, so the container runs the image artifact instead of the local source tree.

The backend container sets:

- `DATA_DIR=/app/data`
- `CHROMA_DIR=/app/data/chroma`
- `REQUIRE_TENANT_ID=false` by default; set it to `true` and pass `X-Tenant-ID` for tenant-scoped local chat/run storage.

Secrets can be provided directly through `OPENAI_API_KEY`, `TAVILY_API_KEY`, and `API_KEY`, or through mounted secret files:

```env
OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
TAVILY_API_KEY_FILE=/run/secrets/tavily_api_key
API_KEY_FILE=/run/secrets/backend_api_key
```

Direct env values win over `*_FILE` values. In production, prefer mounted secret files or a platform secret manager over local `.env` files.

Health checks call:

```text
GET /api/v1/health
```

The health payload includes app status, version, and whether runtime data paths are configured and present.

## Observability

For trace correlation across clients, proxies, and the backend, pass W3C `traceparent` on requests. The backend returns `traceparent`, `X-Trace-ID`, and `X-Span-ID`, and includes trace IDs in JSON request logs and streaming chat events.

To export FastAPI spans to an OTLP HTTP collector:

```env
OTEL_ENABLED=true
OTEL_SERVICE_NAME=ai-research-assistant-backend
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces
```

## Production Gaps

- Add a real process manager or platform-level autoscaling strategy if running more than one backend instance.
- Move tenant-scoped JSON file storage and in-memory rate limiting to external services for multi-instance deployments.
- Use platform-managed secrets or mounted secret files for API keys; avoid baking secrets into images or compose files.
- Run live evaluation on an indexed production-like paper corpus before publishing benchmark claims.
- Audit tracked runtime artifacts and logs before publishing images or source snapshots.
- Provision and operate the OpenTelemetry collector/backend outside this repo.
