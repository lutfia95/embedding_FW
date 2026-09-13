from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import FrameworkError
from .pipeline import Pipeline
from .registry import default_registry
from .serialization import request_from_dict


def main() -> None:
    parser = argparse.ArgumentParser(prog="plms-framework")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("models", help="List registered models and availability")
    for command in ("estimate", "embed", "compare"):
        child = subparsers.add_parser(command)
        child.add_argument("request", help="JSON request filename, or - for stdin")
    serve = subparsers.add_parser("serve", help="Start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    try:
        if args.command == "models":
            result = {
                "models": [
                    {**manifest.to_dict(), "availability": adapter.availability().__dict__}
                    for manifest, adapter in default_registry.models()
                ]
            }
        elif args.command == "serve":
            try:
                import uvicorn
            except ImportError as exc:
                raise SystemExit("Install API dependencies: pip install '.[api]'") from exc
            uvicorn.run("plms_framework.api:app", host=args.host, port=args.port)
            return
        else:
            text = sys.stdin.read() if args.request == "-" else Path(args.request).read_text()
            domain = request_from_dict(json.loads(text), compare=args.command == "compare")
            pipeline = Pipeline()
            result = (
                pipeline.estimate(domain)
                if args.command == "estimate"
                else getattr(pipeline, args.command)(domain)
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    except FrameworkError as exc:
        print(json.dumps(exc.as_dict(), indent=2), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
