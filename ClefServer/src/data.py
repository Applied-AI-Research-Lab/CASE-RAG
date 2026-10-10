from __future__ import annotations

import ast
import csv
import gzip
import io
import json
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

VERIF_LABELS = {"SUPPORTS": "supported", "REFUTES": "refuted",
                "NOT ENOUGH INFO": "notEnoughInfo"}


@dataclass
class Query:
    id: str
    dataset: str
    split: str
    question: str
    label: str
    gold_evidence: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _zip_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import zipfile as _zf
        with _zf.ZipFile(path) as zf:
            _ = zf.namelist()
        return True
    except Exception:
        return False


def fever_examples(path: Path, split: str) -> Iterator[dict]:
    for i, row in enumerate(_read_jsonl(path)):
        label = VERIF_LABELS.get(row.get("label", ""), "notEnoughInfo")
        yield {
            "id": f"fever-{row.get('id', i)}",
            "dataset": "fever",
            "split": split,
            "question": row["claim"],
            "label": label,
            "gold_evidence": [e[2] for e in row.get("evidence", []) if e and len(e) > 2],
            "meta": {"verifiable": row.get("verifiable", "")},
        }


def fever_corpus_from_beir(zip_path: Path, out_dir: Path) -> int:
    out_file = out_dir / "fever" / "corpus.jsonl"
    if out_file.exists():
        return sum(1 for _ in _read_jsonl(out_file))
    out_dir.joinpath("fever").mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(zip_path)
    inner = [n for n in zf.namelist() if n.endswith("corpus.jsonl")]
    assert inner, f"corpus.jsonl not found inside {zip_path}"
    rows: list[dict] = []
    with zf.open(inner[0]) as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({
                "id": d["_id"],
                "dataset": "fever",
                "title": d.get("title", ""),
                "text": d.get("text", ""),
            })
    return _write_jsonl(out_file, rows)


def scifact_load_claims(data_dir: Path, split: str,
                        corpus_lookup: dict | None = None) -> Iterator[dict]:
    path = data_dir / "claims_{s}.jsonl".format(s=split)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            ev = d.get("evidence") or {}
            evidence_texts: list[str] = []
            labels: set[str] = set()
            for doc_id, evs in ev.items():
                for e in evs:
                    lab = e.get("label")
                    if lab:
                        labels.add(lab)

                    if corpus_lookup and doc_id in corpus_lookup:
                        sents = corpus_lookup[doc_id]
                        for sid in e.get("sentences", []):
                            if 0 <= sid < len(sents):
                                evidence_texts.append(sents[sid])
            if "SUPPORT" in labels:
                label = "supported"
            elif "CONTRADICT" in labels:
                label = "refuted"
            else:
                label = "notEnoughInfo"
            yield {
                "id": f"scifact-{d['id']}",
                "dataset": "scifact",
                "split": split,
                "question": d["claim"],
                "label": label,
                "gold_evidence": evidence_texts,
                "meta": {"cited_doc_ids": d.get("cited_doc_ids", [])},
            }


def scifact_corpus_lookup(data_dir: Path) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    with (data_dir / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            abstract = d.get("abstract") or ""
            if isinstance(abstract, str) and abstract.startswith("["):
                try:
                    abstract = ast.literal_eval(abstract)
                except (ValueError, SyntaxError):
                    pass
            if isinstance(abstract, list):
                sents = [str(p) for p in abstract]
            else:
                sents = str(abstract).split(". ")
            lookup[str(d["doc_id"])] = sents
    return lookup


def scifact_corpus(data_dir: Path, out_dir: Path) -> int:
    out_file = out_dir / "scifact" / "corpus.jsonl"
    if out_file.exists():
        return sum(1 for _ in _read_jsonl(out_file))
    out_dir.joinpath("scifact").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with (data_dir / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            abstract = d.get("abstract") or ""

            if isinstance(abstract, str) and abstract.startswith("["):
                try:
                    parts = ast.literal_eval(abstract)
                    abstract = " ".join(str(p) for p in parts)
                except (ValueError, SyntaxError):
                    pass
            text = str(abstract).strip()
            if text:
                rows.append({
                    "id": f"{d['doc_id']}",
                    "dataset": "scifact",
                    "title": d.get("title", ""),
                    "text": text,
                })

                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
                for j, s in enumerate(sents):
                    rows.append({
                        "id": f"{d['doc_id']}-s{j}",
                        "dataset": "scifact",
                        "title": d.get("title", ""),
                        "text": s,
                    })
    return _write_jsonl(out_file, rows)


def freshqa_rows(parquet_path: Path, split: str) -> Iterator[dict]:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    for i, (_, r) in enumerate(df.iterrows()):
        golds = []
        for j in range(10):
            v = r.get(f"answer_{j}")
            if v is not None and str(v).strip():
                golds.append(str(v).strip())
        yield {
            "id": f"freshqa-{split}-{r.get('id', i)}",
            "dataset": "freshqa",
            "split": split,
            "question": str(r.get("question", "")),
            "label": "answerable",
            "gold_evidence": [],
            "meta": {
                "answers": golds,
                "answer_type": str(r.get("fact_type", "")),
                "false_premise": bool(r.get("false_premise", False)),
                "effective_year": str(r.get("effective_year", "")),
            },
        }


def hotpotqa_corpus(parquet_path: Path, out_dir: Path) -> int:
    import pandas as pd
    out_file = out_dir / "hotpotqa" / "corpus.jsonl"
    if out_file.exists():
        return sum(1 for _ in _read_jsonl(out_file))
    out_dir.joinpath("hotpotqa").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    rows: list[dict] = []
    for i, r in df.iterrows():
        ctx = r.get("context") or {}
        titles = list(ctx.get("title", [])) if isinstance(ctx, dict) else []
        sents = [list(s) for s in (list(ctx.get("sentences", []))
                                   if isinstance(ctx, dict) else [])]
        for t, ss in zip(titles, sents):
            text = " ".join(str(s) for s in ss).strip()
            if text:
                rows.append({"id": f"hp-{i}-{t}", "dataset": "hotpotqa",
                             "title": str(t), "text": text})

    seen: set[str] = set()
    dedup: list[dict] = []
    for row in rows:
        key = row["title"] + "|" + row["text"][:200]
        if key not in seen:
            seen.add(key)
            dedup.append(row)
    return _write_jsonl(out_file, dedup)


def hotpotqa_rows(parquet_path: Path, split: str) -> Iterator[dict]:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    for i, r in df.iterrows():
        ctx = r.get("context") or {}
        titles = list(ctx.get("title", [])) if isinstance(ctx, dict) else []
        sents = list(ctx.get("sentences", [])) if isinstance(ctx, dict) else []
        sents = [list(s) for s in sents]
        gold = []
        sup = r.get("supporting_facts") or {}
        for t, sid in zip(list(sup.get("title", [])), list(sup.get("sent_id", []))):
            try:
                j = titles.index(t)
                if sid < len(sents[j]):
                    gold.append(sents[j][sid])
            except (ValueError, IndexError):
                continue
        answers = r.get("answer") or ""
        answers = [str(answers)] if not isinstance(answers, list) else [str(a) for a in answers]
        yield {
            "id": f"hotpot-{split}-{i}",
            "dataset": "hotpotqa",
            "split": split,
            "question": str(r.get("question", "")),
            "label": "answerable",
            "gold_evidence": gold,
            "meta": {"answers": answers, "level": str(r.get("level", ""))},
        }


def prepare_all() -> None:
    splits_dir = OUT / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, int] = {}


    fever_dir = RAW / "fever"
    if (fever_dir / "train.jsonl").exists() and _zip_ok(fever_dir / "fever_beir.zip"):
        report["fever_train"] = _write_jsonl(
            splits_dir / "fever_train.jsonl",
            fever_examples(fever_dir / "train.jsonl", "train"))
        report["fever_dev"] = _write_jsonl(
            splits_dir / "fever_dev.jsonl",
            fever_examples(fever_dir / "shared_task_dev.jsonl", "dev"))
        report["fever_corpus"] = fever_corpus_from_beir(
            fever_dir / "fever_beir.zip", OUT / "corpus")


    scifact_dir = RAW / "scifact" / "data"
    if not (scifact_dir / "claims_train.jsonl").exists():
        tar_path = RAW / "scifact" / "scifact.tar.gz"
        if tar_path.exists():
            import tarfile as _tar
            with _tar.open(tar_path, "r:gz") as tf:
                tf.extractall(RAW / "scifact")
    if (scifact_dir / "claims_train.jsonl").exists():
        lookup = scifact_corpus_lookup(scifact_dir)
        report["scifact_train"] = _write_jsonl(
            splits_dir / "scifact_train.jsonl",
            scifact_load_claims(scifact_dir, "train", lookup))
        report["scifact_dev"] = _write_jsonl(
            splits_dir / "scifact_dev.jsonl",
            scifact_load_claims(scifact_dir, "dev", lookup))
        report["scifact_corpus"] = scifact_corpus(scifact_dir, OUT / "corpus")


    freshqa_dir = RAW / "freshqa"
    if (freshqa_dir / "dev.parquet").exists():
        report["freshqa_dev"] = _write_jsonl(
            splits_dir / "freshqa_dev.jsonl",
            freshqa_rows(freshqa_dir / "dev.parquet", "dev"))
        report["freshqa_test"] = _write_jsonl(
            splits_dir / "freshqa_test.jsonl",
            freshqa_rows(freshqa_dir / "test.parquet", "test"))


    hotpot_dir = RAW / "hotpotqa"
    if (hotpot_dir / "validation.parquet").exists():
        report["hotpotqa_dev"] = _write_jsonl(
            splits_dir / "hotpotqa_dev.jsonl",
            hotpotqa_rows(hotpot_dir / "validation.parquet", "dev"))
        report["hotpotqa_corpus"] = hotpotqa_corpus(
            hotpot_dir / "validation.parquet", OUT / "corpus")

    print(json.dumps(report, indent=2))
    if not report:
        raise SystemExit("No raw data found. Run scripts/download_datasets.sh first.")


if __name__ == "__main__":
    prepare_all()
