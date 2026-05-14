#!/usr/bin/env python3
"""
Serve the simulation (scripts/) on http://127.0.0.1:8765 and accept POST /api/save-capture
to write view and/or displacement PNGs under DATA_DIR, optionally under a subfolder per batch run.

Flat writes (no "folder" in JSON) are capped at CAPTURE_LIMIT POSTs per server run (each POST may include view only, disp only, or both).

Writes with a sanitized "folder" field go under DATA_DIR/<folder>/ with no cap.
"""
from __future__ import annotations

import base64
import json
import re
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(
    r"C:\Users\pdanahy3\OneDrive - Georgia Institute of Technology\Documents\GT_Research_Spring-2026\ACADIA-2026\Research\ACADIA-2026\data"
)

CAPTURE_LIMIT = 100
_saved_pairs_flat = 0
_flat_lock = threading.Lock()

_FOLDER_SAFE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}$")


def _safe_subfolder(name: str) -> str | None:
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if not name or not _FOLDER_SAFE.match(name) or ".." in name:
        return None
    return name


class CaptureHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SCRIPT_DIR), **kwargs)

    def end_headers(self) -> None:
        # SharedArrayBuffer + sim workers need a cross-origin isolated context (Chrome).
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def do_POST(self) -> None:
        global _saved_pairs_flat
        if self.path != "/api/save-capture":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            view_b64 = payload.get("viewPng")
            disp_b64 = payload.get("dispPng")
            if view_b64 is None and disp_b64 is None:
                raise ValueError("payload must include viewPng and/or dispPng")
            legacy_idx = payload.get("index")
            view_idx = int(payload["viewIndex"]) if "viewIndex" in payload else (
                int(legacy_idx) if legacy_idx is not None else 0
            )
            disp_idx = int(payload["dispIndex"]) if "dispIndex" in payload else (
                int(legacy_idx) if legacy_idx is not None else 0
            )
            view_bytes = base64.b64decode(view_b64) if view_b64 is not None else None
            disp_bytes = base64.b64decode(disp_b64) if disp_b64 is not None else None
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        sub = _safe_subfolder(payload.get("folder", ""))
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if sub:
            out_dir = DATA_DIR / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            if view_bytes is not None:
                (out_dir / f"view_{view_idx:06d}.png").write_bytes(view_bytes)
            if disp_bytes is not None:
                (out_dir / f"disp_{disp_idx:06d}.png").write_bytes(disp_bytes)
        else:
            with _flat_lock:
                if _saved_pairs_flat >= CAPTURE_LIMIT:
                    self.send_response(403)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"ok": False, "error": "capture_limit", "limit": CAPTURE_LIMIT}
                        ).encode("utf-8")
                    )
                    return
                out_dir = DATA_DIR
                out_dir.mkdir(parents=True, exist_ok=True)
                if view_bytes is not None:
                    (out_dir / f"view_{view_idx:06d}.png").write_bytes(view_bytes)
                if disp_bytes is not None:
                    (out_dir / f"disp_{disp_idx:06d}.png").write_bytes(disp_bytes)
                _saved_pairs_flat += 1

        with _flat_lock:
            flat_saved = _saved_pairs_flat

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "ok": True,
                    "path": str(out_dir),
                    "flat_saved": flat_saved,
                }
            ).encode("utf-8")
        )

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), format % args)
        )


def main() -> None:
    port = 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), CaptureHTTPRequestHandler)
    print(f"Serving {SCRIPT_DIR}")
    print(f"Open http://127.0.0.1:{port}/sheet-metal-simulation.html or batch.html")
    print(f"Captures -> {DATA_DIR} (flat cap {CAPTURE_LIMIT} POSTs; subfolder writes uncapped)")
    print("Threaded HTTP: concurrent /api/save-capture requests supported.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
