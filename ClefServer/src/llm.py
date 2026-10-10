from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import requests

HF_ROUTER_URL = "https://router.huggingface.co/v1"


@dataclass
class LLMBackend:
    backend: str = "ollama"
    model: str = "gemma4:31b-cloud"
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 512


    disable_thinking: bool = False
    max_retries: int = 3
    latency_s: list[float] = field(default_factory=list)
    n_calls: int = 0
    usage: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = {
                "ollama": "http://localhost:11434/v1",
                "vllm": "http://localhost:8000/v1",
                "hf_router": HF_ROUTER_URL,
            }.get(self.backend, "http://localhost:8000/v1")
        if not self.api_key:
            if self.backend == "hf_router":
                self.api_key = os.environ.get("HF_TOKEN", "")
            elif self.backend == "ollama":
                self.api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        if self.backend in ("vllm", "openai") and not self.base_url.endswith("/v1"):
            self.base_url = self.base_url.rstrip("/") + "/v1"


    def generate(self, prompt: str, max_tokens: int | None = None,
                 system: str | None = None) -> str:
        t0 = time.perf_counter()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                out = self._generate(prompt, max_tokens=max_tokens, system=system)
                self.latency_s.append(time.perf_counter() - t0)
                self.n_calls += 1
                return out
            except (RuntimeError, requests.RequestException) as err:
                last_err = err
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        self.latency_s.append(time.perf_counter() - t0)
        self.n_calls += 1
        raise last_err if last_err else RuntimeError("LLM call failed")

    def _generate(self, prompt: str, max_tokens: int | None = None,
                  system: str | None = None) -> str:
        if self.backend in ("ollama", "vllm", "openai", "hf_router"):
            return self._chat_api(prompt, max_tokens, system)
        if self.backend == "hf":
            return self._hf(prompt, max_tokens)
        raise ValueError(f"unknown backend: {self.backend}")


    def _chat_api(self, prompt: str, max_tokens: int | None = None,
                  system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }


        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        resp = requests.post(
            os.path.join(self.base_url, "chat/completions"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            json=body, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        self.usage = data.get("usage", self.usage)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""

        if not content.strip() and msg.get("reasoning"):
            content = msg["reasoning"]
        return content

    def _hf(self, prompt: str, max_tokens: int | None = None) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not hasattr(self, "_tk"):
            self._tk = AutoTokenizer.from_pretrained(self.model)
            self._mdl = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype="auto", device_map="auto")
        msgs = [{"role": "user", "content": prompt}]
        ids = self._tk.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt").to(self._mdl.device)
        with torch.no_grad():
            out = self._mdl.generate(ids, max_new_tokens=max_tokens or self.max_tokens,
                                     do_sample=False)
        return self._tk.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


    def stats(self) -> dict:
        lat = self.latency_s
        return {
            "n_calls": self.n_calls,
            "mean_latency_s": sum(lat) / len(lat) if lat else 0.0,
            "p50_latency_s": sorted(lat)[len(lat) // 2] if lat else 0.0,
            "total_latency_s": sum(lat),
            "usage": self.usage,
        }
