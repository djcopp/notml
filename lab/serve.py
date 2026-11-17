"""Static file server that sends the cross-origin isolation headers onnxruntime-web needs for WASM threads.

Without COOP/COEP the page has no SharedArrayBuffer, and onnxruntime-web quietly falls back to one thread.
"""

from __future__ import annotations

import functools
import http.server
import pathlib
import threading


class IsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def make_server(directory: pathlib.Path, port: int = 8766) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(IsolatedHandler, directory=str(directory))
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve_in_background(directory: pathlib.Path, port: int = 0) -> http.server.ThreadingHTTPServer:
    server = make_server(directory, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
