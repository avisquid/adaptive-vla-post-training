"""HTTP boundary for the collection dashboard: static UI, state polling and commands."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from real.collection.guide import Conflict


ROOT = Path(__file__).resolve().parents[2]
UI = Path(__file__).resolve().parent / "ui"
UI_POLICY = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'"
# The standalone image audit uses embedded styles and photos, with no scripts.
REPORT_POLICY = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; frame-ancestors 'none'"


def make_server(collection, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if args and str(args[0]).startswith("POST"):
                super().log_message(format, *args)

        def send(self, body: bytes, content_type: str, status=200, download=False, cache=False, policy=UI_POLICY):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", policy)
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="collection-session-log.json"')
            self.end_headers()
            self.wfile.write(body)

        def json(self, body: dict, status=200, download=False):
            self.send(json.dumps(body, allow_nan=False).encode(), "application/json; charset=utf-8", status, download)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/state":
                collection.heartbeat(parse_qs(urlsplit(self.path).query).get("client_id", [None])[0])
                return self.json(collection.snapshot())
            if path.startswith("/api/preview/"):
                preview = collection.preview_images.get(path.rsplit("/", 1)[-1])
                if preview:
                    return self.send(preview, "image/jpeg")
                return self.json({"error": "Camera preview unavailable"}, 404)
            if path == "/api/export":
                return self.json(collection.export(), download=True)
            if path == "/image-report":
                report = ROOT / "docs/state_image_report.html"
                if report.is_file():
                    return self.send(report.read_bytes(), "text/html; charset=utf-8", policy=REPORT_POLICY)
                return self.json({"error": "Image report has not been generated"}, 404)
            files = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css", "/favicon.svg": "favicon.svg"}
            if path in files:
                file = UI / files[path]
                return self.send(file.read_bytes(), mimetypes.guess_type(file)[0] or "application/octet-stream")
            if path.startswith("/api/reference/"):
                try:
                    state_id = int(path.rsplit("/", 1)[-1])
                    if not 0 <= state_id < len(collection.task["states"]):
                        raise ValueError()
                    file = Path(collection.task["states"][state_id]["image_path"])
                    return self.send(file.read_bytes(), mimetypes.guess_type(file)[0] or "application/octet-stream", cache=True)
                except (ValueError, TypeError, OSError):
                    return self.json({"error": "Reference image unavailable"}, 404)
            self.json({"error": "Not found"}, 404)

        def do_POST(self):
            if self.path != "/api/command":
                return self.json({"error": "Not found"}, 404)
            # JSON and same-origin requests only; no cross-origin robot actions.
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.json({"error": "Cross-origin action rejected"}, 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.json({"error": "Expected application/json"}, 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid command size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                self.json(collection.command(body))
            except Conflict as exc:
                self.json({"error": str(exc)}, 409)
            except (ValueError, UnicodeError) as exc:
                self.json({"error": str(exc)}, 400)
            except Exception:
                self.log_error("Collector action failed")
                self.json({"error": "The collector could not apply this action; it was not confirmed. Refresh before retrying."}, 500)

    return ThreadingHTTPServer((host, port), Handler)
