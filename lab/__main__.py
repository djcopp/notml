"""Command line: python -m lab {run,web,browser,serve} [case names...]"""

from __future__ import annotations

import argparse
import sys

import experiments
from lab import harness


def _select(names: list[str], small: bool):
    """Names can be experiment modules (``regex_dfa``) or single cases (``regex_dfa_int32``)."""
    modules = [n for n in names if n in experiments.NAMES] or ([] if names else experiments.NAMES)
    cases = [c for m in modules for c in experiments.load(m, small)]
    wanted = set(names) - set(experiments.NAMES)
    if wanted:
        cases += [c for c in experiments.all_cases(small) if c.name in wanted]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m lab")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in [
        ("run", "run cases natively against their baselines"),
        ("web", "bundle cases for the browser bench"),
    ]:
        p = sub.add_parser(name, help=text)
        p.add_argument("names", nargs="*", help="experiments or cases (default: all)")
        p.add_argument("--small", action="store_true", help="small sizes, for a quick check")
    p = sub.add_parser("browser", help="run the bundled cases in headless Chromium")
    p.add_argument("names", nargs="*")
    p.add_argument("--ort", default="1.30.0", help="onnxruntime-web version")
    p.add_argument("--reps", type=int, default=7)
    p = sub.add_parser("serve", help="serve the browser bench with cross-origin isolation")
    p.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if args.command == "run":
        for case in _select(args.names, args.small):
            harness.run_case(case)
    elif args.command == "web":
        from lab import web

        web.export(_select(args.names, args.small))
    elif args.command == "browser":
        from lab import browser

        browser.run(args.ort, args.names or None, args.reps)
    elif args.command == "serve":
        from lab.serve import make_server
        from lab.web import WEB

        print(f"http://127.0.0.1:{args.port}/index.html")
        make_server(WEB, args.port).serve_forever()


if __name__ == "__main__":
    main()
