from __future__ import annotations

import argparse
import math
import threading
import time

from .routing.clef_runtime import load_clef, score_record


def _softmax(d: dict[str, float]) -> dict[str, float]:
    m = max(d.values())
    e = {k: math.exp(v - m) for k, v in d.items()}
    s = sum(e.values()) or 1.0
    return {k: v / s for k, v in e.items()}


def build_app(model, processor, jsm, max_length: int, served_name: str):
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title=served_name)
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": served_name}

    @app.post("/v1/decide")
    def decide(record: dict = Body(...)) -> dict:
        t0 = time.perf_counter()
        try:
            with lock:
                logits, n_tokens = score_record(model, processor, jsm, record,
                                                max_length=max_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "model": served_name,
            "logits": logits,
            "probabilities": {q: _softmax(v) for q, v in logits.items()},
            "usage": {"input_tokens": n_tokens, "output_tokens": 0,
                      "latency_s": round(time.perf_counter() - t0, 4)},
        }

    @app.post("/v1/systemone")
    def systemone(request: dict = Body(...)) -> dict:
        request.setdefault("model", served_name)
        try:
            with lock:
                return jsm.systemone(model, processor, request, max_length=max_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Cloudflare/clef")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--served-model-name", default="clef")
    args = ap.parse_args()

    import uvicorn

    t0 = time.perf_counter()
    print(f"[clef] loading {args.model} ...", flush=True)
    model, processor, jsm = load_clef(args.model, device="cuda")
    print(f"[clef] loaded in {time.perf_counter() - t0:.0f}s; "
          f"serving on http://{args.host}:{args.port}", flush=True)
    app = build_app(model, processor, jsm, args.max_length, args.served_model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
