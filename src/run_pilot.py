from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from .calibration import (ConformalRiskControl, TemperatureScaling,
                           expected_calibration_error, brier_score)
from .llm import LLMBackend
from .retrieval import build_retriever
from .routing import (ACTIONS, EvidenceState, FeatureRouter,
                      FixedBudgetRouter, LLMRouter)
from .routing.jev_adapter import JevRouter, JevUnavailable
from .routing.selfrag_router import SelfRAGRouter
from .verifier import build_verifier

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "splits"
CORPUS = ROOT / "data" / "processed" / "corpus"
RUNS = ROOT / "runs"

VERIF_PROMPT = """You are a fact verifier. Given the claim and evidence, classify:

- supported : the evidence clearly supports the claim
- refuted   : the evidence clearly contradicts the claim
- notEnoughInfo : the evidence does not decide the claim

Claim: {claim}
Evidence:
{evidence}

Respond with exactly one word: supported, refuted, or notEnoughInfo."""

QA_PROMPT = """Answer the question using ONLY the evidence. If the evidence is
insufficient, answer with exactly: CANNOT_ANSWER.

Question: {question}
Evidence:
{evidence}

Answer (be concise):"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classify_correct(row: dict, pred: str, answer: str) -> bool:
    if row["dataset"] in ("fever", "scifact"):
        return extract_label(pred) == row["label"].strip().lower()
    golds = row.get("meta", {}).get("answers") or []
    if not golds:
        return False
    norm = lambda s: " ".join(str(s).lower().split())
    n_ans = norm(answer)
    if n_ans in ("cannot_answer", "cannot answer", "n/a", "unknown"):
        return False
    return any(norm(g) in n_ans for g in golds if norm(g))


VERIF_LABELS = ("notenoughinfo", "supported", "refuted")


def extract_label(pred: str) -> str:
    t = " ".join(str(pred).lower().split())
    if t in VERIF_LABELS:
        return t
    best, pos = "", -1
    for lab in VERIF_LABELS:
        i = t.rfind(lab)
        if i > pos:
            best, pos = lab, i
    return best


def _ckpt_path(out: str, phase: str) -> Path:
    return Path(f"{out}.{phase}.ckpt.jsonl")


def _ckpt_fingerprint(args) -> str:
    import hashlib
    keys = ("dataset", "split", "n", "calib", "calib_split", "k",
            "max_retrievals", "corpus_file", "corpus_max", "retriever",
            "verifier", "router", "backend", "llm_model", "seed",
            "answer_max_tokens", "answer_system", "force_steps")
    blob = "|".join(f"{k}={getattr(args, k, None)!r}" for k in keys)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_ckpt(out: str, phase: str, fingerprint: str) -> dict:
    p = _ckpt_path(out, phase)
    done: dict = {}
    if not p.exists():
        return done
    lines = p.read_text().splitlines()
    if not lines:
        return done
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError:
        header = {}
    if header.get("_fingerprint") != fingerprint:
        print(f"[{phase}] checkpoint is from a different configuration; ignoring it")
        p.unlink()
        return done
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("id") is not None:
            done[rec["id"]] = rec
    return done


def evaluate_split(rows, evaluate_row, out: str, phase: str, args=None) -> list:
    fp = _ckpt_fingerprint(args) if args is not None else "none"
    done = _load_ckpt(out, phase, fp)
    if done:
        print(f"[{phase}] resuming: {len(done)}/{len(rows)} already evaluated")
    ck = _ckpt_path(out, phase)
    new_file = not ck.exists()
    results = []
    with ck.open("a") as fh:
        if new_file:
            fh.write(json.dumps({"_fingerprint": fp}) + "\n")
            fh.flush()
        for row in rows:
            rid = row.get("id")
            if rid in done:
                results.append(done[rid])
                continue
            rec = evaluate_row(row)
            results.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    return results


def candidate_confidence_prompt(row: dict, evidence: list[dict], answer: str) -> float:
    return 0.5


def run_split(args: argparse.Namespace) -> dict:
    t_start = time.time()


    split_file = SPLITS / f"{args.dataset}_{args.split}.jsonl"
    rows_all = load_jsonl(split_file)


    random.Random(args.seed).shuffle(rows_all)
    test_rows = rows_all[:args.n]
    if args.calib_split and args.calib_split != args.split:
        calib_all = load_jsonl(SPLITS / f"{args.dataset}_{args.calib_split}.jsonl")
        random.Random(args.seed + 1).shuffle(calib_all)
        calib_rows = calib_all[:args.calib]
    else:
        calib_rows = rows_all[args.n:args.n + args.calib]
    print(f"[data] {args.dataset}: test={args.split}(n={len(test_rows)}) "
          f"calib={args.calib_split or args.split}(n={len(calib_rows)})")


    corpus_file = (Path(args.corpus_file) if args.corpus_file
                   else CORPUS / args.dataset / "corpus.jsonl")
    passages = load_jsonl(corpus_file)
    if args.corpus_max:
        passages = passages[:args.corpus_max]
    print(f"[corpus] {len(passages)} passages")


    if args.retriever == "none" or not passages:
        from .retrieval import NoneRetriever
        retriever = NoneRetriever()
    else:
        retriever = build_retriever(args.retriever, passages,
                                    dense_model=args.dense_model, device=args.device)
    verifier = build_verifier(args.verifier, device=args.device,
                              model_name=args.verifier_model)
    llm = LLMBackend(backend=args.backend, model=args.llm_model,
                     base_url=args.base_url or LLMBackend(backend=args.backend).base_url,
                     api_key=args.api_key or None)

    if args.router == "feature":
        router = FeatureRouter()
    elif args.router == "fixed":
        router = FixedBudgetRouter(force_steps=args.force_steps)
    elif args.router == "llm":
        router = LLMRouter(llm)
    elif args.router == "selfrag":
        router = SelfRAGRouter(llm=llm)
    elif args.router == "jev":
        router = JevRouter()
        if not router.available():
            raise JevUnavailable("TYPESAFE_API_KEY not set; use --router feature|llm|selfrag")
    else:
        raise ValueError(f"unknown router: {args.router}")


    def evaluate_row(row: dict) -> dict:
        t_row = time.perf_counter()
        state = EvidenceState(query=row["question"], max_retrievals=args.max_retrievals)
        per_step_retrievals = 0
        decision = None
        t_retrieval = t_llm = t_verifier = t_router = 0.0
        n_evidence_chars = 0
        for step in range(args.max_retrievals + 1):
            t0 = time.perf_counter()
            depth = args.k * (step + 1)
            hits = retriever.rank(state.query, k=depth)
            new_ids = {h.passage_id for h in hits}
            have = {p["id"] for p in state.passages}
            for p in passages:
                if p["id"] in new_ids and p["id"] not in have:
                    state.passages.append(p)
                    have.add(p["id"])
                    n_evidence_chars += len(p.get("text", ""))
            t_retrieval += time.perf_counter() - t0
            per_step_retrievals += 1
            state.retrieval_steps = step
            state.scores = [h.score for h in hits]


            if row["dataset"] in ("fever", "scifact"):
                prompt = VERIF_PROMPT.format(
                    claim=row["question"],
                    evidence="\n".join(f"- {p.get('text','')[:1500]}" for p in state.passages))
            else:
                prompt = QA_PROMPT.format(
                    question=row["question"],
                    evidence="\n".join(f"- {p.get('text','')[:1500]}" for p in state.passages))
            t0 = time.perf_counter()
            answer = llm.generate(prompt, max_tokens=args.answer_max_tokens,
                                  system=args.answer_system or None)
            t_llm += time.perf_counter() - t0
            state.candidate_answer = answer.strip()


            t0 = time.perf_counter()
            state.verifier_scores = verifier.score_batch(
                row["question"], state.passages)[:args.k]
            t_verifier += time.perf_counter() - t0
            state.candidate_confidence = (
                max(state.verifier_scores) if state.verifier_scores else 0.0)

            t0 = time.perf_counter()
            decision = router.decide(state)
            t_router += time.perf_counter() - t0
            if decision.action == "ANSWER":
                break
            if decision.action == "ABSTAIN" or decision.action == "ESCALATE":
                break

        else:
            decision = router.decide(state)

        correct = classify_correct(row, state.candidate_answer, state.candidate_answer)
        rec = {
            "id": row["id"],
            "label": row["label"],
            "answer": state.candidate_answer,
            "action": decision.action,
            "probabilities": decision.probabilities,
            "evidence_sufficiency": decision.evidence_sufficiency,
            "estimated_factual_risk": decision.estimated_factual_risk,
            "n_retrievals": per_step_retrievals,
            "n_evidence_passages": len(state.passages),
            "n_evidence_chars": n_evidence_chars,
            "correct": bool(correct),
            "timing": {
                "total_s": time.perf_counter() - t_row,
                "retrieval_s": t_retrieval,
                "llm_s": t_llm,
                "verifier_s": t_verifier,
                "router_s": t_router,
            },
        }
        if hasattr(router, "stats"):
            rs = router.stats()
            rec["router_stats"] = {"n_calls": rs.get("n_calls", 0),
                                   "mean_latency_s": rs.get("mean_latency_s", 0.0)}
        return rec


    print("[calib] running calibration split ...")
    t_calib = time.perf_counter()
    calib_out = evaluate_split(calib_rows, evaluate_row, args.out, "calib", args)
    calib_time_s = time.perf_counter() - t_calib
    llm_calls_after_calib = llm.stats()["n_calls"]
    calib_conf = [o["probabilities"]["ANSWER"] for o in calib_out]
    calib_correct = [float(o["correct"]) for o in calib_out]

    from .calibration import IsotonicCalibrator
    temp = TemperatureScaling().fit(calib_conf, calib_correct)
    iso = IsotonicCalibrator().fit(calib_conf, calib_correct)

    if iso.report and temp.report and iso.report.ece <= temp.report.ece:
        calibrator, cal_method = iso, "isotonic"
    else:
        calibrator, cal_method = temp, "temperature"

    calib_conf_cal = [calibrator(c) for c in calib_conf]
    conformal = ConformalRiskControl(target_risk=args.target_risk).fit(
        calib_conf_cal, calib_correct)
    print(f"[calib] method={cal_method} temp={temp.temperature:.3f} "
          f"iso_ece={iso.report.ece:.3f} temp_ece={temp.report.ece:.3f} "
          f"conformal_threshold="
          f"{'none (nothing certifiable)' if conformal.threshold is None else f'{conformal.threshold:.3f}'}")


    print("[test] running evaluation split ...")
    t_test = time.perf_counter()
    test_out = evaluate_split(test_rows, evaluate_row, args.out, "test", args)
    test_time_s = time.perf_counter() - t_test
    llm_calls_after_test = llm.stats()["n_calls"]
    conf = [calibrator(o["probabilities"]["ANSWER"]) for o in test_out]
    correct = [o["correct"] for o in test_out]


    decisions = []
    for o, c in zip(test_out, conf):
        if o["action"] in ("ABSTAIN", "ESCALATE"):
            decisions.append(o["action"])
        elif conformal.should_answer(c):
            decisions.append("ANSWER")
        else:
            decisions.append("ABSTAIN")

    from .evaluation import (abstention_metrics, area_over_risk_coverage,
                             risk_coverage_curve, selective_risk_at_coverage)

    answered_mask = [d == "ANSWER" for d in decisions]
    answered_conf = [c for c, m in zip(conf, answered_mask) if m]
    answered_correct = [float(x) for x, m in zip(correct, answered_mask) if m]

    results = {
        "config": vars(args),
        "n_calib": len(calib_out),
        "n_test": len(test_out),
        "raw_accuracy": sum(correct) / len(correct) if correct else 0.0,
        "conformal_decisions": {
            "coverage": sum(answered_mask) / len(answered_mask) if answered_mask else 0,
            "answered_risk": (1 - sum(answered_correct) / len(answered_correct)
                              if answered_correct else 0.0),
        },
        "metrics": {
            "ece": expected_calibration_error(conf, np_float(correct)),
            "brier": brier_score(conf, np_float(correct)),
            "aurc": area_over_risk_coverage(conf, np_float(correct)),
            "risk_at_80_cov": selective_risk_at_coverage(
                conf, np_float(correct), 0.80),
            "abstention": abstention_metrics(decisions, correct),
        },
        "cost": {
            "mean_retrievals": sum(o["n_retrievals"] for o in test_out) / len(test_out),
            "mean_evidence_passages": (sum(o["n_evidence_passages"] for o in test_out)
                                       / len(test_out)),
            "mean_evidence_chars": (sum(o["n_evidence_chars"] for o in test_out)
                                    / len(test_out)),
            "llm_calls": llm.stats()["n_calls"],
            "llm_calls_calib": llm_calls_after_calib,
            "llm_calls_test": llm_calls_after_test - llm_calls_after_calib,
            "mean_llm_latency_s": llm.stats()["mean_latency_s"],
            "p50_llm_latency_s": llm.stats()["p50_latency_s"],
            "total_llm_latency_s": llm.stats()["total_latency_s"],
            "llm_usage": llm.stats().get("usage", {}),
        },
        "timing": {
            "calib_phase_s": calib_time_s,
            "test_phase_s": test_time_s,
            "total_runtime_s": time.time() - t_start,
            "per_query_test": {
                key: {
                    "mean_s": _mean([o["timing"][key] for o in test_out]),
                    "p50_s": _p50([o["timing"][key] for o in test_out]),
                    "total_s": sum(o["timing"][key] for o in test_out),
                }
                for key in ("total_s", "retrieval_s", "llm_s", "verifier_s", "router_s")
            },
            "router_calls_test": sum(o.get("router_stats", {}).get("n_calls", 0)
                                      for o in test_out),
            "router_mean_latency_s": _mean(
                [o.get("router_stats", {}).get("mean_latency_s", 0.0)
                 for o in test_out if o.get("router_stats")]),
        },
        "calibration": {"method": cal_method,
                        "temperature": temp.temperature,
                        "report": asdict(temp.report)},
        "conformal": {"target_risk": args.target_risk,
                      "threshold": conformal.threshold},
        "runtime_s": time.time() - t_start,
        "records": test_out,
        "calib_records": calib_out,
        "risk_coverage_curve": risk_coverage_curve(conf, np_float(correct)),
    }
    return results


def np_float(x):
    import numpy as np
    return np.asarray([1.0 if v else 0.0 for v in x], dtype=float)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _p50(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[len(s) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fever", "scifact",
                                                          "freshqa", "hotpotqa"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--calib", type=int, default=150)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-retrievals", type=int, default=2)
    ap.add_argument("--corpus-max", type=int, default=50000)
    ap.add_argument("--corpus-file", default=None,
                    help="override corpus path (e.g. FEVER gold-page corpus)")
    ap.add_argument("--retriever", default="bm25",
                    choices=["bm25", "dense", "hybrid", "none"])
    ap.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--verifier", default="nli",
                    choices=["nli", "embedding", "none"])
    ap.add_argument("--verifier-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--router", default="feature",
                    choices=["feature", "fixed", "llm", "jev", "selfrag"])
    ap.add_argument("--answer-max-tokens", type=int, default=64,
                    help="token budget for the answer call (default 64)")
    ap.add_argument("--answer-system", default="",
                    help="system prompt for the answer call; needed by reasoning "
                         "backbones that otherwise emit chain-of-thought instead "
                         "of the requested label")
    ap.add_argument("--force-steps", type=int, default=None,
                    help="router=fixed: retrieval rounds to force (default: full budget)")
    ap.add_argument("--backend", default="ollama",
                    choices=["ollama", "vllm", "openai", "hf_router", "hf"])
    ap.add_argument("--llm-model", default="gemma4:31b-cloud")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--target-risk", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the calibration/test split (exchangeability)")
    ap.add_argument("--calib-split", default=None,
                    help="split file for calibration (default: same as --split); "
                         "use 'train' to keep small dev sets fully for testing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = run_split(args)
    out_path = Path(args.out) if args.out else (
        RUNS / f"pilot_{args.dataset}_{args.split}_{args.router}_{args.retriever}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in results.items() if k != "records"}
    slim["risk_coverage_curve"] = results["risk_coverage_curve"]
    slim["calib_records"] = results["calib_records"]
    out_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False))
    records_path = out_path.with_suffix(".records.json")
    records_path.write_text(json.dumps(results["records"], indent=2, ensure_ascii=False))

    print(json.dumps({k: v for k, v in slim.items()
                      if k in ("raw_accuracy", "metrics", "cost", "conformal_decisions",
                               "calibration", "conformal")},
                     indent=2))
    print(f"\n[out] {out_path}\n[out] {records_path}")


if __name__ == "__main__":
    main()
