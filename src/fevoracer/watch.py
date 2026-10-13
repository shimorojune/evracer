"""`uv run watch`: watch training live in the browser, from a separate process.

Serves a page that polls the trainer's snapshot file (logs/live/*.json)
every few seconds. Nothing here touches the training processes.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

import fevo

from .paths import LOGS


@lru_cache(maxsize=None)
def _track_json(track_id: str) -> bytes:
    t = fevo.load_track(track_id).raw
    ref = fevo.reference(track_id)
    return json.dumps({"id": track_id, "name": fevo.load_track(track_id).name, "centerline": t["centerline"], "widthLeft": t["widthLeft"],
                       "widthRight": t["widthRight"], "gates": t["gates"], "racingOffset": ref["offset"], "referenceLapS": ref["lapS"]}).encode()  # fmt: skip


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # quiet
        pass

    def _send(self, body: bytes, kind: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("content-type", kind)
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(resources.files("fevoracer").joinpath("watch.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/runs":
            runs = sorted(p.stem for p in (LOGS / "live").glob("*.json")) if (LOGS / "live").is_dir() else []
            return self._send(json.dumps(runs).encode(), "application/json")
        if path.startswith("/live/") and path.endswith(".json"):
            name = path[len("/live/") :]
            f = LOGS / "live" / name
            if "/" not in name and f.is_file():
                return self._send(f.read_bytes(), "application/json")
        if path.startswith("/track/") and path.endswith(".json"):
            tid = path[len("/track/") : -len(".json")]
            if tid in fevo.track_ids():
                return self._send(_track_json(tid), "application/json")
        self._send(b"not found", "text/plain", 404)


def _is_watch_server(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/runs", timeout=0.5) as r:
            return r.headers.get("content-type", "").startswith("application/json")
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(prog="watch", description="Watch training live (separate process; doesn't slow training)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    server = None
    for port in range(a.port, a.port + 20):
        if _is_watch_server(port):
            url = f"http://localhost:{port}/"
            print(f"A watch page is already running at {url}; opening it.")
            if not a.no_open:
                webbrowser.open(url)
            return
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue  # port taken by something else: try the next one
    if server is None:
        raise SystemExit(f"No free port between {a.port} and {a.port + 19}; pass --port.")
    url = f"http://localhost:{server.server_address[1]}/"
    print(f"Watching training at {url} (Ctrl+C to stop). Training itself isn't affected.")
    if not a.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
