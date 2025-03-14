"""OpenTelemetry tracer initialization.

When OTEL_EXPORTER_OTLP_ENDPOINT is unset, init_tracing returns False and
the default NoOpTracerProvider makes every span call a zero-cost no-op.
"""
from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def init_tracing(service_name: str) -> bool:
    """Configure the global tracer provider. Returns True if enabled."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        return False

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    return True


def instrument_app(app) -> None:
    """Auto-instrument FastAPI handlers and httpx outbound calls."""
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


def get_tracer(name: str):
    return trace.get_tracer(name)
