"""Local live judge-log viewer and optional standalone HTML exports."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TEMPLATE = Path(__file__).resolve().parent.parent / "judge-viewer.html"


def model_metadata(source: Path) -> dict:
    # Never read the current .config: it contains secrets and may differ from this run.
    for manifest in [source.parent / "run.json", source.with_suffix(".summary.json")]:
        if manifest.is_file():
            try:
                settings = json.loads(manifest.read_text()).get("settings", {})
                return {
                    "policy_model": settings.get("training", {}).get("model", ""),
                    "judge_model": settings.get("judge", {}).get("model", ""),
                    **(
                        {"score_name": settings["judge"]["score_name"]}
                        if "score_name" in settings.get("judge", {}) else {}
                    ),
                }
            except (ValueError, OSError, AttributeError):
                # A manifest can be mid-write while the training process starts.
                continue
    return {}


def find_logs(root: Path) -> list[dict]:
    logs = []
    if not root.is_dir():
        return logs
    for source in root.rglob("*.jsonl"):
        if source.name not in {"judge.jsonl", "comparison.jsonl"} and not source.name.endswith(
            ".judge.jsonl"
        ):
            continue
        try:
            if not source.resolve().is_relative_to(root) or not source.is_file():
                continue
            stat = source.stat()
        except OSError:
            continue
        logs.append(
            {
                "path": source.relative_to(root).as_posix(),
                "version": f"{stat.st_mtime_ns}:{stat.st_size}",
                "modified": stat.st_mtime_ns,
            }
        )
    return sorted(logs, key=lambda row: row["modified"], reverse=True)


def read_live_log(source: Path) -> dict:
    text = source.read_bytes().decode("utf-8", errors="replace")
    pending = False
    # A poll can catch the final JSON record mid-write. Keep all complete records
    # visible and pick up the unfinished one on the next poll.
    if text and not text.endswith("\n"):
        last_break = text.rfind("\n") + 1
        try:
            json.loads(text[last_break:])
        except ValueError:
            text = text[:last_break]
            pending = True
    return {"text": text, "pending": pending, "metadata": model_metadata(source)}


def make_viewer_server(root: str | Path = "runs", port: int = 8765) -> ThreadingHTTPServer:
    root = Path(root).expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, value, status=HTTPStatus.OK, html=False):
            data = value.encode() if html else json.dumps(value).encode()
            self.send_response(status)
            self.send_header(
                "Content-Type", "text/html; charset=utf-8" if html else "application/json"
            )
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    self.reply(
                        TEMPLATE.read_text().replace("<body>", '<body data-live="true">', 1),
                        html=True,
                    )
                elif request.path == "/api/logs":
                    self.reply({"logs": find_logs(root)})
                elif request.path == "/api/log":
                    name = parse_qs(request.query).get("path", [""])[0]
                    allowed = {row["path"] for row in find_logs(root)}
                    source = (root / name).resolve()
                    if name not in allowed or not source.is_relative_to(root):
                        self.reply({"error": "Judge log not found"}, HTTPStatus.NOT_FOUND)
                        return
                    self.reply({"name": name, **read_live_log(source)})
                else:
                    self.reply({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except OSError:
                self.reply(
                    {"error": "Log temporarily unavailable; retrying on the next refresh"},
                    HTTPStatus.NOT_FOUND,
                )

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_viewer(root: str | Path = "runs", port: int = 8765) -> None:
    with make_viewer_server(root, port) as server:
        print(f"Judge viewer: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Watching {Path(root).expanduser().resolve()} — Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def export_viewer(log: str | Path, output: str | Path | None = None) -> Path:
    source = Path(log).expanduser().resolve()
    target = Path(output).expanduser().resolve() if output else source.with_suffix(".html")
    if target == source or target.suffix.lower() != ".html":
        raise ValueError("Viewer output must be an .html file distinct from the input log")
    text = source.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("The judge log is empty")
    metadata = model_metadata(source)
    payload = json.dumps({"name": source.name, "text": text, "metadata": metadata})
    # JSON is placed inside a non-executable script tag. Escape HTML delimiters so
    # even logged '</script>' text cannot break out of it. The UI uses textContent.
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    html = TEMPLATE.read_text().replace(
        '<script id="embedded-log" type="application/json">null</script>',
        f'<script id="embedded-log" type="application/json">{payload}</script>',
        1,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target
