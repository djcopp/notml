"""Static server with cross-origin isolation headers (COOP/COEP) so onnxruntime-web can use WASM threads."""

import functools
import http.server
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


if __name__ == "__main__":
    port, directory = int(sys.argv[1]), sys.argv[2]
    http.server.ThreadingHTTPServer(
        ("", port), functools.partial(Handler, directory=directory)
    ).serve_forever()
