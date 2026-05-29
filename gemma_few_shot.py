"""
HW4 gemma-4-31b-it (NIM API) + few-shot in-context examples。

基於 api_llm_inference.py，加上跟 qwen_few_shot.py 同樣的 few-shot prompt
組裝邏輯。Prompt + 範例選擇與 qwen_few_shot.py 完全一致；差異只在後端
(NIM gemma vs local Qwen GGUF)。
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as _dt
import json
import os
import re
import sys
import threading
import time
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm.auto import tqdm

# 重用 qwen_few_shot.py 的 few-shot prompt builder + parser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_few_shot import (  # noqa: E402
    build_prompt,
    extract_answer_letter,
    _FEWSHOT_IDS,
    _EXAMPLES_CACHE,
)
import qwen_few_shot  # for setting _FEWSHOT_IDS / _EXAMPLES_CACHE


# ---------------------------------------------------------------------------
# Console / logging (copy from api_llm_inference)
# ---------------------------------------------------------------------------

_LOG_LOCK = threading.Lock()


def _configure_stdout() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def log(msg: str) -> None:
    with _LOG_LOCK:
        try:
            tqdm.write(msg)
        except UnicodeEncodeError:
            sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: "collections.deque[float]" = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                wait_until = self._timestamps[0] + self.window
            sleep_for = max(0.05, wait_until - time.monotonic())
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# NIM API
# ---------------------------------------------------------------------------

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "google/gemma-4-31b-it"


def load_api_keys(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        log(f"[FATAL] 找不到 API Key 檔案 '{filepath}'"); sys.exit(1)
    keys: List[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                keys.append(stripped)
    if not keys:
        log(f"[FATAL] API Key 檔案空"); sys.exit(1)
    return keys


def query_llm(prompt: str, api_key: str, model_name: str, timeout: float,
              temperature: float, top_p: float) -> str:
    """NIM API + assistant prefix `Answer: ` + max_tokens=4 + stop=["\n"]"""
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Answer: "},
        ],
        "max_tokens": 4,
        "temperature": temperature,
        "top_p": top_p,
        "stop": ["\n"],
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        if "choices" in d and d["choices"]:
            content = d["choices"][0].get("message", {}).get("content", "")
            return content.strip() if content else "[Error] 空白回應"
        return "[Error] 無 choices"
    except requests.exceptions.RequestException as exc:
        return f"[Error] API 請求失敗: {exc}"
    except Exception as exc:
        return f"[Error] 其他: {exc}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_existing(path: str) -> Dict[int, dict]:
    if not os.path.exists(path):
        return {}
    try:
        return {int(r["id"]): r for r in json.load(open(path)) if "id" in r}
    except Exception:
        return {}


def save_outputs(submission_csv: str, backup_json: str,
                 rows: List[dict], results: Dict[int, dict]) -> None:
    pd.DataFrame(rows).to_csv(submission_csv + ".tmp", index=False)
    os.replace(submission_csv + ".tmp", submission_csv)
    with open(backup_json + ".tmp", "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
    os.replace(backup_json + ".tmp", backup_json)


def _build_rows(samples: List[dict], results: Dict[int, dict]) -> List[dict]:
    rows = []
    for s in samples:
        sid = s["id"]
        letter = (results.get(sid) or {}).get("predicted_answer") or ""
        if not letter:
            letter = sorted((s.get("options") or {}).keys())[:1] or ["A"]
            letter = letter[0]
        rows.append({"id": sid, "answer": letter})
    return rows


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def process_sample(sample: dict, api_key: str, rate_limiter: RateLimiter,
                   model_name: str, max_attempts: int,
                   temperature: float, top_p: float,
                   previous: Optional[dict] = None) -> dict:
    sid = sample["id"]
    options = sample.get("options", {}) or {}
    valid = set(options.keys())
    prompt = build_prompt(sample.get("full_context", ""),
                          sample.get("current_step", ""),
                          options)
    best = (previous or {}).get("predicted_answer", "") or ""
    last_resp = (previous or {}).get("model_raw_response", "")
    if best in valid:
        return {"id": sid, "current_step": sample.get("current_step", ""),
                "prompt_mode": "fewshot_gemma",
                "predicted_answer": best, "model_raw_response": last_resp}
    for attempt in range(1, max_attempts + 1):
        rate_limiter.acquire()
        timeout = 60.0 + 30.0 * attempt
        resp = query_llm(prompt, api_key, model_name, timeout, temperature, top_p)
        last_resp = resp
        letter = extract_answer_letter(resp, valid)
        if letter:
            best = letter
            break
        if resp.startswith("[Error]"):
            log(f"[WARN] id={sid} a{attempt}: {resp[:120].replace(chr(10),' ')}")
        time.sleep(min(2.0 * attempt, 20.0))
    return {"id": sid, "current_step": sample.get("current_step", ""),
            "prompt_mode": "fewshot_gemma",
            "predicted_answer": best, "model_raw_response": last_resp}


def run_pipeline(input_path, output_csv, backup_json, api_key_path, model_name,
                 max_attempts, rate, workers_per_key,
                 temperature, top_p, save_every):
    api_keys = load_api_keys(api_key_path)
    samples = load_jsonl(input_path)
    log(f"載入 {len(samples)} 筆樣本 / 使用模型 {model_name}")
    existing = load_existing(backup_json)
    if existing:
        log(f"沿用既有 {len(existing)} 筆")
    results: Dict[int, dict] = dict(existing)

    to_process = []
    skipped = 0
    for s in samples:
        sid = s["id"]
        prev = results.get(sid)
        plet = (prev or {}).get("predicted_answer", "")
        pmod = (prev or {}).get("prompt_mode", "")
        if plet in set((s.get("options") or {}).keys()) and pmod == "fewshot_gemma":
            skipped += 1
        else:
            to_process.append(s)

    n_keys = len(api_keys)
    total_workers = workers_per_key * n_keys
    log(f"沿用 {skipped} / 重打 {len(to_process)} | {n_keys} keys × {workers_per_key} = {total_workers} threads")
    rate_limiters = [RateLimiter(max_calls=rate) for _ in api_keys]

    if not to_process:
        save_outputs(output_csv, backup_json, _build_rows(samples, results), results)
        return

    results_lock = threading.Lock()
    save_lock = threading.Lock()
    completed = 0
    total = len(to_process)

    def _worker(s: dict, key_idx: int) -> str:
        sid = s["id"]
        with results_lock:
            prev = results.get(sid)
        try:
            r = process_sample(
                s, api_keys[key_idx], rate_limiters[key_idx], model_name,
                max_attempts, temperature, top_p, previous=prev,
            )
        except Exception as exc:
            log(f"[WARN] id={sid} exc: {exc}")
            r = prev or {"id": sid, "current_step": s.get("current_step", ""),
                         "prompt_mode": "fewshot_gemma",
                         "predicted_answer": "", "model_raw_response": ""}
        with results_lock:
            results[sid] = r
        return r.get("predicted_answer", "") or ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as ex, \
            tqdm(total=len(samples), initial=skipped, desc="gemma_fs", unit="q",
                 dynamic_ncols=True) as bar:
        futs = {ex.submit(_worker, s, i % n_keys): s
                for i, s in enumerate(to_process)}
        for fut in concurrent.futures.as_completed(futs):
            completed += 1
            try:
                fut.result()
            except Exception as exc:
                log(f"[WARN] fut: {exc}")
            bar.update(1)
            if completed % save_every == 0 or completed == total:
                with save_lock, results_lock:
                    rows = _build_rows(samples, results)
                    try:
                        save_outputs(output_csv, backup_json, rows, results)
                    except Exception as exc:
                        log(f"[WARN] flush: {exc}")

    with save_lock, results_lock:
        save_outputs(output_csv, backup_json, _build_rows(samples, results), results)
    log(f"完成 → {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HW4 gemma-4-31b-it NIM few-shot")
    p.add_argument("--input", default=os.path.join("data", "test.jsonl"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--backup", default="llm_answering_results.json")
    p.add_argument("--api-key", default="api_key.txt")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--max-attempts", type=int, default=8)
    p.add_argument("--rate", type=int, default=37)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.1)
    p.add_argument("--fewshot-ids", default=",".join(str(x) for x in _FEWSHOT_IDS))
    return p.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()
    parsed_ids = [int(x) for x in re.split(r"[,;\s]+", args.fewshot_ids.strip()) if x]
    if parsed_ids != _FEWSHOT_IDS:
        qwen_few_shot._FEWSHOT_IDS = parsed_ids
        qwen_few_shot._EXAMPLES_CACHE = None
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, args.output)
    backup_json = os.path.join(args.output_dir, args.backup)
    log(f"輸出: {args.output_dir} | few-shot ids: {qwen_few_shot._FEWSHOT_IDS}")
    run_pipeline(
        args.input, output_csv, backup_json, args.api_key, args.model,
        args.max_attempts, args.rate, args.workers,
        args.temperature, args.top_p, args.save_every,
    )


if __name__ == "__main__":
    main()
