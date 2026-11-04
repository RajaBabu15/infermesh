"""Shared test helpers: a gateway wired to a fake HTTP upstream, and small stubs."""
import httpx
import pytest

from infermesh.main import create_app
from infermesh.settings import GatewayConfig

CHAT = {"model": "m", "messages": [{"role": "system", "content": "shared system prompt " * 5},
                                   {"role": "user", "content": "hi"}]}


class DummyLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


def cfg(**kw):
    return GatewayConfig(_env_file=None, **kw)


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    """Build the real app whose workers talk to `handler` over an httpx MockTransport.

    Extra keyword arguments become INFERMESH_* settings.
    """
    def build(handler, workers=("w1",), **env):
        path = tmp_path / "workers.yaml"
        path.write_text("workers:\n" + "".join(
            f"  - {{id: {w}, type: openai, url: 'http://{w}/v1', model: m, api_key_env: NO_KEY}}\n"
            for w in workers))
        monkeypatch.setenv("INFERMESH_WORKERS_CONFIG_PATH", str(path))
        for key, value in env.items():
            monkeypatch.setenv(f"INFERMESH_{key.upper()}", str(value))
        app = create_app()
        app.state.proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return app
    return build


def asgi_client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


def ok_response(request, content="hello"):
    return httpx.Response(200, json={
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 2,
                  "prompt_tokens_details": {"cached_tokens": 64}},
    })
