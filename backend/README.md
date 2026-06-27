# AI Research Assistant Backend

FastAPI backend skeleton for the AI Research Assistant MVP.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

