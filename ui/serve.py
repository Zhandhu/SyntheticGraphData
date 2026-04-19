"""Serve the QA + Knowledge Graph viewer UI with a generation API.

Endpoints:
  GET  /ui/viewer.html      — the viewer
  GET  /data/output/*       — static data files (ui_data.json, etc.)
  POST /api/generate        — trigger QA generation at a given complexity
  GET  /api/status          — check whether a generation job is running
"""

import asyncio
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from urllib.parse import urlparse

PORT = 8765
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ── Generation job state ──────────────────────────────────────────────────────
_job_lock = threading.Lock()
_job: dict = {"running": False, "pid": None, "started": None,
               "complexity": None, "log": []}


def _run_generation(complexity: float | None, batches: int, batch_size: int, api_key: str | None = None):
    """Run generate_until_pass.py in a subprocess and stream output to _job."""
    global _job
    cmd = [sys.executable, "scripts/generate_until_pass.py",
           "--batches", str(batches), "--batch-size", str(batch_size)]
    if complexity is not None:
        cmd += ["--complexity", str(complexity)]

    env = os.environ.copy()
    # Source env.sh values if present
    env_sh = os.path.join(ROOT, "configs", "env.sh")
    if os.path.exists(env_sh):
        with open(env_sh) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    parts = line[7:].split("=", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        val = parts[1].strip().strip('"').strip("'")
                        if val and not val.startswith("#"):
                            env[k] = val
    # API key supplied via UI takes priority
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=ROOT,
    )
    with _job_lock:
        _job["pid"] = proc.pid

    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        with _job_lock:
            _job["log"] = lines[-200:]   # keep last 200 lines
    proc.wait()

    with _job_lock:
        _job["running"] = False
        _job["pid"] = None


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def do_POST(self):
        if self.path == "/api/generate":
            self._handle_generate()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self):
        if self.path == "/api/status":
            self._handle_status()
        else:
            super().do_GET()

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_generate(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        with _job_lock:
            if _job["running"]:
                self._json({"error": "A generation job is already running."}, 409)
                return
            _job["running"] = True
            _job["started"] = time.time()
            _job["log"] = []
            _job["complexity"] = body.get("complexity")

        complexity = body.get("complexity")   # None = original pipeline
        batches    = int(body.get("batches", 5))
        batch_size = int(body.get("batch_size", 8))
        api_key    = body.get("api_key") or None

        t = threading.Thread(
            target=_run_generation,
            args=(complexity, batches, batch_size, api_key),
            daemon=True,
        )
        t.start()
        self._json({"status": "started", "complexity": complexity,
                    "batches": batches, "batch_size": batch_size})

    def _handle_status(self):
        with _job_lock:
            snap = dict(_job)
        elapsed = (time.time() - snap["started"]) if snap["started"] else None
        self._json({
            "running": snap["running"],
            "complexity": snap["complexity"],
            "elapsed_s": round(elapsed, 1) if elapsed else None,
            "log_tail": snap["log"][-20:],
        })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    url = f"http://localhost:{PORT}/ui/viewer.html"
    print(f"Serving at {url}")
    webbrowser.open(url)
    with http.server.HTTPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
