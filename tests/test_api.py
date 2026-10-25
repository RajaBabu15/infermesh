"""OpenAI surface + control endpoints via in-process ASGI (no upstream)."""
import asyncio
import os
from pathlib import Path

import httpx

HERE = Path(__file__).parent


def _client():
    os.environ["INFERMESH_WORKERS_CONFIG_PATH"] = str(HERE / "_test_workers.yaml")
    from infermesh.main import create_app
    app = create_app()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def test_core_endpoints_without_upstream():
    async def go():
        async with _client() as c:
            h = await c.get("/health")
            assert h.status_code == 200
            assert h.json()["status"] == "ok"

            models = await c.get("/v1/models")
            assert "test-model" in [m["id"] for m in models.json()["data"]]

            # unknown model -> 404 model_not_found, as OpenAI does (no upstream call)
            r = await c.post("/v1/chat/completions",
                             json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 404
            assert r.json()["error"]["code"] == "model_not_found"

            mt = await c.get("/metrics")
            assert "infermesh_requests_total" in mt.text
            assert "infermesh_workers_healthy" in mt.text

    asyncio.run(go())
