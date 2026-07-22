# Security Notes

The project ingests untrusted data from PDFs, web search results, and downloaded URLs. This file documents implemented mitigations and remaining gaps.

## Implemented

PDF download:

- only `http` and `https` URLs are allowed
- optional `PDF_DOWNLOAD_ALLOWED_DOMAINS` blocks PDF downloads outside configured domains
- localhost, private, link-local, multicast, reserved, and unspecified IPs are blocked after DNS resolution
- `Content-Length` is checked before streaming when available
- streamed bytes are capped by `MAX_PDF_DOWNLOAD_BYTES`
- non-PDF content is rejected by content type and PDF header checks
- partial temp files are removed on failure

Prompt injection:

- retrieved context is explicitly described as untrusted data in the answer prompt
- suspicious instruction-like chunks are marked with `security_flag: suspicious_instruction`
- patterns include previous-instruction override, system/developer prompt references, role override, and secret exfiltration
- `draft_answer` records `suspicious_context_count` in trace when such chunks are present

API/data:

- optional API key authentication protects API endpoints when `API_KEY` is set
- `X-API-Key` and `Authorization: Bearer <key>` are accepted for protected requests
- optional tenant enforcement is enabled with `REQUIRE_TENANT_ID=true`; protected requests must then include a valid `X-Tenant-ID`
- tenant-aware requests echo `X-Tenant-ID` and local chat/run storage is scoped under `data/tenants/<tenant_id>/`
- optional in-memory request limiting is enabled with `API_RATE_LIMIT_PER_MINUTE`
- rate-limit buckets are scoped by tenant id plus API key or client host when `X-Tenant-ID` is present
- rate-limited responses include `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`
- API responses include `X-Request-ID`, `traceparent`, `X-Trace-ID`, and `X-Span-ID`, preserving caller-provided IDs for audit/debug correlation
- request completion logs are JSON and include request ID, trace ID, span ID, method, path, status, and latency
- health, docs, OpenAPI, Redoc, and CORS preflight requests remain public
- local PDF uploads are capped by `MAX_PDF_UPLOAD_BYTES`
- uploaded files must use a `.pdf` filename and start with the `%PDF` header
- filename path traversal is reduced with basename and safe filename handling
- chat/web-ingested chunks are scoped by `chat_id` where applicable
- `.gitignore` blocks local `.env` files and runtime data while allowing `.env.example` and `.gitkeep` placeholders

## Remaining Gaps

- API key auth plus `X-Tenant-ID` is still coarse-grained and not a substitute for OAuth/user sessions
- in-memory rate limiting is per process and not suitable for multi-instance deployments; use Redis/API gateway in production
- OpenTelemetry instrumentation is available, but collector/backend provisioning is deployment infrastructure
- no antivirus or sandboxed PDF parsing
- no full semantic prompt-injection classifier
- local JSON storage has tenant namespaces, but a production multi-user deployment should still use a database with tenant/user isolation and backups
- existing tracked runtime artifacts should be audited before publishing the repository

## Recommended Next Steps

1. Add OAuth/user sessions and role-based tenant authorization.
2. Move rate limiting to Redis/API gateway for multi-instance deployments.
3. Add antivirus or sandboxed PDF parsing for production uploads.
4. Add a stronger classifier for malicious retrieved context.
5. Move storage to a database with tenant/user isolation.
6. Remove or sanitize tracked runtime artifacts before making the repo public.
