"""Langfuse tracing: records what each query really did (every LLM, embedding, search and index step).

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set. Spans are sent in the background, so a
slow or unreachable Langfuse server does not slow down a query. Steps are only recorded inside trace().
"""
import logging
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_client = None
_failed = False
_in_trace: ContextVar[bool] = ContextVar("in_trace", default=False)

# SerpAPI takes its key as a URL parameter, and httpx puts the URL in its error messages.
_URL_SECRET = re.compile(r"((?:api_?key|access_token|token|secret)=)[^&\s'\"]+", re.I)


def enabled() -> bool:
    return bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY) and not _failed


def _secrets() -> list[str]:
    vals = (config.API_KEY, config.EMBED_API_KEY, config.SERPAPI_KEY, config.S2_API_KEY,
            config.JWT_SECRET, config.LANGFUSE_SECRET_KEY)
    return [v for v in vals if len(v) >= 8]


def redact(data):
    """Remove API keys from anything that is about to leave the machine."""
    if isinstance(data, str):
        for s in _secrets():
            data = data.replace(s, "[redacted]")
        return _URL_SECRET.sub(r"\1[redacted]", data)
    if isinstance(data, dict):
        return {k: redact(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [redact(v) for v in data]
    return data


def client():
    """The shared Langfuse client, or None when tracing is off."""
    global _client, _failed
    if not enabled():
        return None
    with _lock:
        if _client is None:
            try:
                from langfuse import Langfuse
                from opentelemetry.sdk.trace import TracerProvider
                # Own provider: a global one would also switch on FastAPI's built-in telemetry.
                _client = Langfuse(tracer_provider=TracerProvider(), public_key=config.LANGFUSE_PUBLIC_KEY,
                                   secret_key=config.LANGFUSE_SECRET_KEY, base_url=config.LANGFUSE_BASE_URL,
                                   flush_interval=2, mask=lambda *, data, **_: redact(data))
            except Exception:
                _failed = True
                log.exception("Langfuse could not start, tracing is off")
                return None
    return _client


def shutdown() -> None:
    if _client is not None:
        _client.shutdown()  # sends whatever is still queued


class _Noop:
    def update(self, **_):
        return self


_NOOP = _Noop()


@contextmanager
def _observe(lf, name: str, as_type: str, **kw):
    error = None
    with lf.start_as_current_observation(name=name, as_type=as_type, **kw) as obs:
        try:
            yield obs
        except Exception as e:
            # Raised again below, outside the span: otherwise OpenTelemetry stores the raw
            # message and stack trace, which are not redacted.
            obs.update(level="ERROR", status_message=redact(f"{type(e).__name__}: {e}")[:500])
            error = e
    if error is not None:
        raise error


@contextmanager
def span(name: str, as_type: str = "span", **kw):
    """One step of a query. Does nothing when tracing is off or no trace is active."""
    lf = client() if _in_trace.get() else None
    if lf is None:
        yield _NOOP
        return
    with _observe(lf, name, as_type, **kw) as obs:
        yield obs


@contextmanager
def trace(name: str, user: str, input: dict, session: str | None = None):
    """One user query, from the API request to the response."""
    lf = client()
    if lf is None:
        yield _NOOP
        return
    from langfuse import propagate_attributes
    from opentelemetry import context
    token = _in_trace.set(True)
    parent = context.attach(context.Context())  # always a new trace, whatever headers the request carried
    try:
        with propagate_attributes(user_id=user, session_id=session, trace_name=name, tags=["paper-discovery"]):
            with _observe(lf, name, "chain", input=input) as obs:
                yield obs
    finally:
        context.detach(parent)
        _in_trace.reset(token)
