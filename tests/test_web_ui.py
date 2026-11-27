"""Web UI tests: static checks on the page, plus an optional headless-browser run (ScriptedLLM, no key)."""

import json
import re
import socket
import threading
import time
from contextlib import contextmanager

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dolphinmind.agent import Document, ScriptedLLM  # noqa: E402
from dolphinmind.evaluation import NEEDLES, insert_needle, make_haystack  # noqa: E402
from dolphinmind.server import create_app  # noqa: E402
from dolphinmind.server.app import STATIC_DIR  # noqa: E402

BOOK = insert_needle(make_haystack(3_000), NEEDLES[0]["text"], 60)
CID = next(i for i, c in enumerate(Document(BOOK).chunks) if "ALPHA7392BETA" in c)
REPLIES = [
    json.dumps({"thought": "look for it", "tool": "grep", "args": {"pattern": "authentication code", "k": 1}}),
    json.dumps({"thought": "found it", "answer": "ALPHA7392BETA", "citations": [CID]}),
]


def _page() -> str:
    return TestClient(create_app(llm_factory=lambda: ScriptedLLM(REPLIES))).get("/").text


def test_index_served_as_html():
    client = TestClient(create_app(llm_factory=lambda: ScriptedLLM(REPLIES)))
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.text == (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "<title>DolphinAgent</title>" in r.text
    assert "/" not in client.get("/openapi.json").json()["paths"]  # UI is not part of the API schema


def test_page_is_self_contained_and_calls_existing_routes():
    page = _page()
    # no CDN or other external fetches: the demo must work offline
    assert not re.search(r"""(src|href)=["']https?://""", page)
    assert "import(" not in page
    # relative URLs, so the UI also works when the API is mounted under a path prefix
    assert 'fetch("docs"' in page and "new EventSource(\"ask/stream?\"" in page
    assert "fetch(`docs/${encodeURIComponent(state.docId)}/chunks/${cid}`)" in page
    paths = TestClient(create_app()).get("/openapi.json").json()["paths"]
    assert {"/docs", "/ask/stream", "/docs/{doc_id}/chunks/{chunk_id}"} <= set(paths)
    # model output and document text are untrusted: never rendered as HTML
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page and "document.write" not in page


@contextmanager
def _serve(app):
    uvicorn = pytest.importorskip("uvicorn")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.02)
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.should_exit = True
        thread.join(10)


@pytest.fixture
def browser_page():
    sync_api = pytest.importorskip("playwright.sync_api")
    try:
        pw = sync_api.sync_playwright().start()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"playwright unavailable: {e}")
    try:
        browser = pw.chromium.launch()
    except Exception as e:  # noqa: BLE001 - no browser binary installed
        pw.stop()
        pytest.skip(f"chromium unavailable: {e}")
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    try:
        yield page, errors
    finally:
        browser.close()
        pw.stop()


def test_browser_upload_ask_stream_and_click_citation(browser_page):
    page, errors = browser_page
    with _serve(create_app(llm_factory=lambda: ScriptedLLM(REPLIES))) as url:
        page.goto(url)
        assert page.is_disabled("#ask")  # nothing uploaded yet
        page.fill("#text", BOOK)
        page.click("#upload")
        page.wait_for_selector("#doc-status.ok")
        assert f"{len(Document(BOOK).chunks)} chunks" in page.text_content("#doc-status")

        page.fill("#question", "What is the authentication code?")
        page.press("#question", "Enter")
        page.wait_for_selector("#ask-status.ok")

        steps = page.locator("#steps li")
        assert steps.count() == 1
        assert "grep(" in steps.first.locator(".call").text_content()
        assert page.text_content("#answer") == "ALPHA7392BETA"
        # citation chip opens the chunk with the answer highlighted
        page.wait_for_selector("#chunk mark")
        assert page.text_content("#chunk-title") == f"Chunk {CID}"
        assert page.text_content("#chunk mark") == "ALPHA7392BETA"
        # "[chunk N]" markers in observations are clickable too
        page.locator("#steps .chip").first.click()
        assert page.locator(f"#citations .chip.active[data-chunk='{CID}']").count() == 1
    assert errors == []


def test_browser_reports_backend_error(browser_page):
    page, errors = browser_page
    with _serve(create_app(llm_factory=lambda: ScriptedLLM([]))) as url:
        page.goto(url)
        page.fill("#text", BOOK)
        page.click("#upload")
        page.wait_for_selector("#doc-status.ok")
        page.fill("#question", "q")
        page.click("#ask")
        page.wait_for_selector("#ask-status.err")
        assert "model backend error" in page.text_content("#ask-status")
        assert page.is_enabled("#ask") and page.is_hidden("#answer-box")
    assert errors == []
