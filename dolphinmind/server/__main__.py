"""``python -m dolphinmind.server [--host H] [--port P] [--semantic]``: serve the agent API with uvicorn."""

from __future__ import annotations

import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m dolphinmind.server",
                                description="Serve the DolphinAgent HTTP API (key from DOLPHIN_API_KEY).")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    p.add_argument("--max-docs", type=int, default=32, help="documents kept in the in-memory cache")
    p.add_argument("--semantic", action="store_true",
                   help="index uploads for search_doc (loads sentence-transformers all-MiniLM-L6-v2)")
    args = p.parse_args(argv)

    import uvicorn

    from .app import create_app
    from .store import DocStore

    def embedder():
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    store = DocStore(max_docs=args.max_docs, embedder_factory=embedder if args.semantic else None)
    uvicorn.run(create_app(store=store), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
