"""Deployment packaging checks: Dockerfile, slim server requirements, no heavy imports (no Docker needed)."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _names(path: Path) -> set:
    out = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.add(re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].lower())
    return out


def test_server_requirements_are_a_slim_subset_of_runtime():
    server, runtime = _names(ROOT / "requirements-server.txt"), _names(ROOT / "requirements.txt")
    assert {"fastapi", "uvicorn", "httpx", "numpy"} <= server <= runtime
    assert not server & {"torch", "transformers", "bitsandbytes", "sentence-transformers", "faiss-cpu"}


def test_dockerfile_installs_slim_deps_runs_server_and_holds_no_secrets():
    docker = (ROOT / "Dockerfile").read_text()
    assert "COPY requirements-server.txt" in docker and "pip install -r requirements-server.txt" in docker
    assert "COPY dolphinmind/ dolphinmind/" in docker
    assert '"-m", "dolphinmind.server", "--host", "0.0.0.0"' in docker
    assert re.search(r"^USER (?!root)", docker, re.MULTILINE)  # not running as root
    assert "/health" in docker and "EXPOSE 8000" in docker
    assert not re.search(r"(?i)(ENV|ARG)\s+\S*(KEY|TOKEN|SECRET)", docker)
    ignored = (ROOT / ".dockerignore").read_text().split()
    assert {".git", ".env", "notebooks", "tests"} <= set(ignored)
    assert (ROOT / "dolphinmind" / "server" / "static" / "index.html").exists()  # shipped with the package


def test_server_imports_without_gpu_or_retrieval_stack():
    """The image has no torch/faiss/sentence-transformers: the app must build without them."""
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'torch', 'transformers', 'faiss', 'sentence_transformers',"
        " 'bitsandbytes', 'pandas', 'scipy'}:\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from dolphinmind.server import create_app, ResponseCache\n"
        "from dolphinmind.agent import Tracer\n"
        "create_app(cache=ResponseCache(), tracer=Tracer())\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
