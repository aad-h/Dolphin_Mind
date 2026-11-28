# DolphinAgent HTTP service: web UI, /ask, SSE /ask/stream, /traces.
#   docker build -t dolphinagent .
#   docker run -p 8000:8000 -e DOLPHIN_API_KEY dolphinagent
# Build with --build-arg SEMANTIC=1 (and run with --semantic) to enable search_doc;
# that adds CPU torch + sentence-transformers and downloads the embedder on first use.
FROM python:3.11-slim

ARG SEMANTIC=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt \
 && if [ "$SEMANTIC" = "1" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu torch \
      && pip install faiss-cpu sentence-transformers; \
    fi

COPY dolphinmind/ dolphinmind/

RUN useradd --create-home --uid 10001 dolphin
USER dolphin

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Keys come from the environment at run time (DOLPHIN_API_KEY, DOLPHIN_BASE_URL, DOLPHIN_MODEL).
ENTRYPOINT ["python", "-m", "dolphinmind.server", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--trace"]
