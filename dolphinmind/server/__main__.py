"""``python -m dolphinmind.server [--host H] [--port P] [--semantic] [--trace] [--otel]``: serve the agent API."""

from __future__ import annotations

import argparse
from typing import Any, List, Optional


def otel_exporter() -> Any:
    """An :class:`OpenTelemetryExporter` over a configured SDK ``TracerProvider``.

    Spans go to an OTLP collector when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (needs
    ``opentelemetry-exporter-otlp``), otherwise to the console.
    """
    import os

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    from ..agent.tracing import OpenTelemetryExporter

    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        span_exporter: Any = OTLPSpanExporter()
    else:
        span_exporter = ConsoleSpanExporter()
    service = os.environ.get("OTEL_SERVICE_NAME", "dolphinagent")
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    return OpenTelemetryExporter(provider.get_tracer("dolphinmind"))


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m dolphinmind.server",
                                description="Serve the DolphinAgent HTTP API (key from DOLPHIN_API_KEY).")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    p.add_argument("--max-docs", type=int, default=32, help="documents kept in the in-memory cache")
    p.add_argument("--semantic", action="store_true",
                   help="index uploads for search_doc (loads sentence-transformers all-MiniLM-L6-v2)")
    p.add_argument("--cache-size", type=int, default=256,
                   help="answers kept in the response cache (0 disables it; default: 256)")
    p.add_argument("--cache-ttl", type=float, default=None, help="seconds before a cached answer expires")
    p.add_argument("--trace", action="store_true", help="record per-question traces (GET /traces)")
    p.add_argument("--otel", action="store_true",
                   help="also export traces to OpenTelemetry (OTLP if OTEL_EXPORTER_OTLP_ENDPOINT is set, "
                        "else console); implies --trace")
    args = p.parse_args(argv)
    if args.cache_size < 0:
        p.error("--cache-size must be >= 0")
    if args.cache_ttl is not None and args.cache_ttl <= 0:
        p.error("--cache-ttl must be > 0")

    import uvicorn

    from ..agent.tracing import Tracer
    from .app import create_app
    from .cache import ResponseCache
    from .store import DocStore

    def embedder():
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    store = DocStore(max_docs=args.max_docs, embedder_factory=embedder if args.semantic else None)
    cache = ResponseCache(args.cache_size, ttl=args.cache_ttl) if args.cache_size else None
    tracer = Tracer(exporters=[otel_exporter()] if args.otel else None) if (args.trace or args.otel) else None
    uvicorn.run(create_app(store=store, cache=cache, tracer=tracer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
