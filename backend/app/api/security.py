from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
import re
from time import monotonic

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from app.config.settings import settings


TENANT_ID_HEADER = "X-Tenant-ID"
PUBLIC_PATHS = {
    "/docs",
    "/openapi.json",
    "/redoc",
}
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_BUCKETS: dict[str, list[float]] = {}
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_tenant_id: ContextVar[str | None] = ContextVar("tenant_id", default=None)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int


async def api_key_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if _is_public_request(request):
        return await call_next(request)

    tenant_id = _normalize_tenant_id(request.headers.get(TENANT_ID_HEADER))
    if settings.REQUIRE_TENANT_ID and not tenant_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Missing or invalid tenant id."},
        )
    tenant_token = _tenant_id.set(tenant_id)
    request.state.tenant_id = tenant_id

    expected_key = settings.API_KEY
    provided_key = request.headers.get("x-api-key") or _bearer_token(request.headers.get("authorization"))
    if expected_key and provided_key != expected_key:
        response = JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing API key."},
        )
        _apply_tenant_header(response, tenant_id)
        _tenant_id.reset(tenant_token)
        return response

    try:
        rate_limit = _rate_limit_decision(request)
        if rate_limit and not rate_limit.allowed:
            response = _rate_limit_response(rate_limit)
            _apply_tenant_header(response, tenant_id)
            return response

        response = await call_next(request)
        if rate_limit:
            _apply_rate_limit_headers(response, rate_limit)
        _apply_tenant_header(response, tenant_id)
        return response
    finally:
        _tenant_id.reset(tenant_token)


def _is_public_request(request: Request) -> bool:
    if request.method == "OPTIONS":
        return True
    path = request.url.path
    if path in PUBLIC_PATHS or path.endswith("/health"):
        return True
    return False


def _bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _rate_limit_decision(request: Request) -> RateLimitDecision | None:
    limit = settings.API_RATE_LIMIT_PER_MINUTE
    if limit <= 0:
        return None

    client_id = _rate_limit_client_id(request)
    now = monotonic()
    window_start = now - _RATE_LIMIT_WINDOW_SECONDS
    calls = [timestamp for timestamp in _RATE_BUCKETS.get(client_id, []) if timestamp >= window_start]
    if len(calls) >= limit:
        _RATE_BUCKETS[client_id] = calls
        return RateLimitDecision(
            allowed=False,
            limit=limit,
            remaining=0,
            reset_seconds=_reset_seconds(now, calls),
        )

    calls.append(now)
    _RATE_BUCKETS[client_id] = calls
    return RateLimitDecision(
        allowed=True,
        limit=limit,
        remaining=max(0, limit - len(calls)),
        reset_seconds=_reset_seconds(now, calls),
    )


def _rate_limit_response(decision: RateLimitDecision) -> JSONResponse:
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded."},
    )
    _apply_rate_limit_headers(response, decision)
    response.headers["Retry-After"] = str(decision.reset_seconds)
    return response


def _apply_rate_limit_headers(response: Response, decision: RateLimitDecision) -> None:
    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    response.headers["X-RateLimit-Reset"] = str(decision.reset_seconds)


def current_tenant_id() -> str | None:
    return _tenant_id.get()


def _normalize_tenant_id(value: str | None) -> str | None:
    if not value:
        return None
    tenant_id = value.strip()
    if not _TENANT_ID_RE.fullmatch(tenant_id):
        return None
    return tenant_id


def _rate_limit_client_id(request: Request) -> str:
    tenant_id = current_tenant_id()
    api_key = request.headers.get("x-api-key") or _bearer_token(request.headers.get("authorization"))
    remote_host = request.client.host if request.client else "unknown"
    if tenant_id and api_key:
        return f"tenant:{tenant_id}:key:{api_key}"
    if tenant_id:
        return f"tenant:{tenant_id}:host:{remote_host}"
    return api_key or remote_host


def _apply_tenant_header(response: Response, tenant_id: str | None) -> None:
    if tenant_id:
        response.headers[TENANT_ID_HEADER] = tenant_id


def _reset_seconds(now: float, calls: list[float]) -> int:
    if not calls:
        return int(_RATE_LIMIT_WINDOW_SECONDS)
    oldest_call = min(calls)
    reset_at = oldest_call + _RATE_LIMIT_WINDOW_SECONDS
    return max(1, int(reset_at - now))
