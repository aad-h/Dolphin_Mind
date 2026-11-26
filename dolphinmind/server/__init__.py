"""HTTP service for DolphinAgent (FastAPI + Server-Sent Events)."""

from .app import create_app
from .store import DocStore, StoredDoc, doc_key

__all__ = ["create_app", "DocStore", "StoredDoc", "doc_key"]
