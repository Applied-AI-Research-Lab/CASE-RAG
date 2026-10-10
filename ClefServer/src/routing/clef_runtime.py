from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _release_module(path: Path) -> ModuleType:
    name = "joint_schema_model"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"{name}.py not found in {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_clef(model_id: str = "Cloudflare/clef", device: str = "cuda"):
    from huggingface_hub import snapshot_download

    path = Path(model_id)
    if not path.is_dir():
        path = Path(snapshot_download(model_id))
    jsm = _release_module(path)
    model, processor = jsm.load_release_model(path, device=device)
    return model, processor, jsm


def validate(jsm: ModuleType, record: dict[str, Any]) -> None:
    questions = record.get("questions")
    if "state" not in record:
        raise ValueError("state is required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for qid, q in questions.items():
        if q.get("type") not in jsm.QUESTION_TYPES:
            raise ValueError(f"{qid}: type must be noul, choice, or score")
        if q["type"] != "noul" and not q.get("criteria"):
            raise ValueError(f"{qid}: criteria must not be empty")


def score_record(model, processor, jsm: ModuleType, record: dict[str, Any],
                 max_length: int = 16384) -> tuple[dict[str, dict[str, float]], int]:
    import torch

    validate(jsm, record)
    enc = jsm.encode_record(processor.tokenizer, record, max_length=max_length,
                            processor=processor)
    device = next(model.parameters()).device
    batch = jsm.collate_records([enc], processor.tokenizer.pad_token_id, device)
    with torch.inference_mode():
        out = model(batch)[0]
    logits = {
        q.question_id: dict(zip(q.option_ids, ql.float().tolist()))
        for q, ql in zip(enc.questions, out)
    }
    return logits, len(enc.input_ids)
