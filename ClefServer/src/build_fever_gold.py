from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "corpus" / "fever" / "corpus_gold.jsonl"


def slugify(title: str) -> str:
    s = title.strip().replace(" ", "_")
    for a, b in ((": ", "-COLON-"), ("(", "-LRB-"), (")", "-RRB-"),
                 (",", "-COMMA-"), ("!", "-EXCLAMATION-"), ("?", "-QUESTION-")):
        s = s.replace(a, b)
    return s


def collect_gold_titles() -> set[str]:
    titles: set[str] = set()
    with (RAW / "fever" / "shared_task_dev.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            for group in d.get("evidence", []):
                if not group:
                    continue
                for e in group:
                    if e and len(e) > 2 and e[2]:
                        titles.add(str(e[2]))
                        break
    return titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-distractors", type=int, default=20000)
    args = ap.parse_args()

    gold_titles = collect_gold_titles()
    print(f"gold evidence titles in dev: {len(gold_titles)}")

    zf = zipfile.ZipFile(RAW / "fever" / "fever_beir.zip")
    inner = [n for n in zf.namelist() if n.endswith("corpus.jsonl")][0]

    gold_rows: list[dict] = []
    distractor_rows: list[dict] = []
    total = 0
    with zf.open(inner) as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            if not line.strip():
                continue
            total += 1
            d = json.loads(line)
            pid = d["_id"]
            if pid in gold_titles or slugify(d.get("title", "")) in gold_titles:
                gold_rows.append({"id": pid, "dataset": "fever",
                                  "title": d.get("title", ""),
                                  "text": d.get("text", "")})
            elif len(distractor_rows) < args.n_distractors:
                distractor_rows.append({"id": pid, "dataset": "fever",
                                        "title": d.get("title", ""),
                                        "text": d.get("text", "")})
    print(f"scanned {total} passages; matched gold: {len(gold_rows)}")


    if len(distractor_rows) < args.n_distractors:
        need = args.n_distractors - len(distractor_rows)

        with zf.open(inner) as fh:
            taken = {r["id"] for r in gold_rows} | {r["id"] for r in distractor_rows}
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                if len(distractor_rows) >= args.n_distractors:
                    break
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["_id"] not in taken and len(d.get("text", "")) > 50:
                    distractor_rows.append({"id": d["_id"], "dataset": "fever",
                                            "title": d.get("title", ""),
                                            "text": d.get("text", "")})

    random.Random(42).shuffle(distractor_rows)
    out_rows = gold_rows + distractor_rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} passages ({len(gold_rows)} gold + "
          f"{len(distractor_rows)} distractors) -> {OUT}")


if __name__ == "__main__":
    main()
