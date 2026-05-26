"""
HW4 LLM Tool Calling Agent — submission pipeline.

End-to-end:
  1. 讀 test.jsonl (每筆: id / full_context / current_step / options)
  2. 對每一筆 id 打 LLM API，請模型在 options 中挑出唯一正確的工具 (A..H)
  3. 自動重試直到拿到合法 letter，或耗盡 max_attempts
  4. 寫出 submission.csv (id, answer) 與 llm_answering_results.json (完整 raw 回應)
  5. 每筆即時 flush，可斷點續跑 (再次執行只會重打尚未拿到 letter 的列)

用法：
  python HW4_111550132.py
  python HW4_111550132.py --input data/test.jsonl --output submission.csv \
      --backup llm_answering_results.json --max-attempts 10 \
      --rate 37 --workers 4

設計重點 (沿用 HW3)：
  - Windows cp950 主控台相容：強制 stdout 為 utf-8、純文字訊息
  - 不會因 API timeout / 空回應 / 解析失敗而崩潰；該筆持續重試
  - 多 API key：api_key.txt 一行一把，N 把 key → 自動開 (N × --workers) 條 thread，
    每把 key 各自獨立 60 秒滑動視窗 rate limit (每把 key max --rate calls/min)
  - Resume：偵測既有 backup JSON 與 submission.csv，已拿到 letter 者直接沿用
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


# ---------------------------------------------------------------------------
# Console / encoding helpers
# ---------------------------------------------------------------------------

def _configure_stdout() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _LOG_LOCK:
        try:
            tqdm.write(msg)
        except UnicodeEncodeError:
            sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Rate limiter — sliding window across all worker threads
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: float = 60.0) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls 必須 > 0")
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
# API
# ---------------------------------------------------------------------------

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# HW4 規範：open-weight、≤ 80B。Gemma / Llama-3.1-70B / Qwen3 都合法。
MODEL_NAME = "google/gemma-3-27b-it"


def slugify_model_name(name: str) -> str:
    tail = name.rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", tail).strip("_")
    return slug or "model"


def default_output_dir(model_name: str) -> str:
    stamp = _dt.datetime.now().strftime("%m%d%H%M")
    return os.path.join("outputs", "hw4", f"{slugify_model_name(model_name)}_{stamp}")


def load_api_keys(filepath: str) -> List[str]:
    """從 api_key.txt 讀出所有 API key (一行一把，忽略空行 / # 註解)。"""
    if not os.path.exists(filepath):
        log(f"[FATAL] 找不到 API Key 檔案 '{filepath}'")
        sys.exit(1)
    keys: List[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            keys.append(stripped)
    if not keys:
        log(f"[FATAL] API Key 檔案 '{filepath}' 是空的")
        sys.exit(1)
    return keys


def query_llm(prompt: str, api_key: str, model_name: str, timeout: float) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 0.1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            return content.strip() if content else "[Error] 模型回傳了空白內容"
        return "[Error] API 回應格式不符預期 (無 choices)"
    except requests.exceptions.RequestException as exc:
        return f"[Error] API 請求失敗: {exc}"
    except json.JSONDecodeError:
        return "[Error] 無法解析 API 傳回的 JSON"
    except Exception as exc:  # pragma: no cover
        return f"[Error] 發生未知例外: {exc}"


# ---------------------------------------------------------------------------
# Prompt / parsing
# ---------------------------------------------------------------------------

# 解析優先順序：
#   1) "Answer: X"      ← 我們在 prompt 強制要求的格式
#   2) "<answer>X</answer>"
#   3) markdown 加粗 **X** / 反引號 `X`
#   4) 出現於句首 / 結尾的單一大寫字母
#   5) 任一在 valid_letters 中出現的單獨大寫字母 (取最後一個)
_ANSWER_LINE_RE = re.compile(r"(?i)\banswer\s*[:：=\-]\s*\*{0,2}\(?\s*([A-H])\s*\)?")
_ANSWER_TAG_RE = re.compile(r"(?i)<\s*answer\s*>\s*\(?\s*([A-H])\s*\)?\s*<\s*/\s*answer\s*>")
_MD_BOLD_RE = re.compile(r"\*\*\s*\(?\s*([A-H])\s*\)?\s*\*\*")
_TICK_RE = re.compile(r"`\s*\(?\s*([A-H])\s*\)?\s*`")
_STANDALONE_RE = re.compile(r"(?<![A-Za-z0-9])\(?([A-H])\)?(?![A-Za-z0-9])")


def extract_answer_letter(response_text: str, valid_letters: Set[str]) -> str:
    if not response_text or response_text.startswith("[Error]"):
        return ""
    for regex in (_ANSWER_LINE_RE, _ANSWER_TAG_RE, _MD_BOLD_RE, _TICK_RE):
        for m in regex.finditer(response_text):
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter
    # fallback：抓最後一個出現的合法 standalone letter (模型通常把答案放結尾)
    last: str = ""
    for m in _STANDALONE_RE.finditer(response_text):
        letter = m.group(1).upper()
        if letter in valid_letters:
            last = letter
    return last


def _format_tool(letter: str, tool: dict) -> str:
    name = tool.get("name", "")
    desc = tool.get("description", "")
    args = tool.get("arguments") or {}
    results = tool.get("results") or {}

    def _format_schema(schema: dict) -> str:
        props = (schema or {}).get("properties") or {}
        if not props:
            return "  (none)"
        lines = []
        for key, spec in props.items():
            spec = spec or {}
            ptype = spec.get("type", "any")
            pdesc = spec.get("description", "")
            if pdesc:
                lines.append(f"  - {key} ({ptype}): {pdesc}")
            else:
                lines.append(f"  - {key} ({ptype})")
        return "\n".join(lines)

    parts = [
        f"[{letter}] name: {name}",
        f"    description: {desc}" if desc else "",
        "    arguments:",
        _format_schema(args),
        "    results:",
        _format_schema(results),
    ]
    return "\n".join(p for p in parts if p != "")


def build_prompt(full_context: str, current_step: str, options: Dict[str, dict]) -> str:
    options_block = "\n".join(
        _format_tool(letter, options[letter]) for letter in sorted(options.keys())
    )
    valid_letters = ", ".join(sorted(options.keys()))
    return (
        "You are an expert tool-calling agent. Given the historical context of a user's task, "
        "the current sub-step to be executed, and a list of candidate tools, your job is to "
        "select the SINGLE correct tool whose name, arguments, and returned results best "
        "match what the current step needs to do.\n\n"
        "Reasoning guidelines:\n"
        "  - Match the tool's *action* (its name + description) to the current step's verb (query / search / book / confirm / etc.).\n"
        "  - Two tools may have similar names (e.g. `search_train` vs `query_past_ticket`); "
        "disambiguate by carefully comparing each tool's argument keys and result keys against "
        "what the current step actually requires.\n"
        "  - Prefer the tool whose required inputs are already available in the full_context.\n"
        "  - The current_step's wording is the strongest signal — only one option will fit exactly.\n\n"
        f"Full context (the user's overall plan so far):\n{full_context}\n\n"
        f"Current step (the sub-task to execute right now):\n{current_step}\n\n"
        f"Candidate tools:\n{options_block}\n\n"
        f"Valid option letters: {valid_letters}\n"
        "Think briefly, then output ONLY the final answer on its own line in this exact format:\n"
        "Answer: <LETTER>\n"
        "where <LETTER> is one of the valid option letters above. Do not output anything after that line."
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        log(f"[FATAL] 找不到輸入檔 '{path}'")
        sys.exit(1)
    samples: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log(f"[WARN] 第 {line_no} 行 JSON 解析失敗，跳過: {exc}")
    return samples


def load_existing_results(path: str) -> Dict[int, dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(r["id"]): r for r in data if "id" in r}
    except Exception as exc:
        log(f"[WARN] 讀取 {path} 失敗，視為空: {exc}")
        return {}


def _atomic_replace(src: str, dst: str, max_retries: int = 8) -> None:
    delay = 0.1
    last_exc: Optional[BaseException] = None
    for _ in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
    raise last_exc  # type: ignore[misc]


def save_outputs(
    output_csv: str,
    backup_json: str,
    submission_rows: List[dict],
    results_by_id: Dict[int, dict],
) -> None:
    tmp_csv = output_csv + ".tmp"
    tmp_json = backup_json + ".tmp"

    pd.DataFrame(submission_rows).to_csv(tmp_csv, index=False)
    _atomic_replace(tmp_csv, output_csv)

    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(list(results_by_id.values()), f, ensure_ascii=False, indent=2)
    _atomic_replace(tmp_json, backup_json)


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def process_sample(
    sample: dict,
    api_key: str,
    rate_limiter: RateLimiter,
    model_name: str,
    max_attempts: int,
    previous_best: Optional[dict] = None,
) -> dict:
    sample_id = sample["id"]
    full_context = sample.get("full_context", "")
    current_step = sample.get("current_step", "")
    options = sample.get("options", {}) or {}
    valid_letters: Set[str] = set(options.keys())
    prompt = build_prompt(full_context, current_step, options)

    best_letter = (previous_best or {}).get("predicted_answer", "") or ""
    last_response = (previous_best or {}).get("model_raw_response", "")

    if best_letter in valid_letters:
        return {
            "id": sample_id,
            "current_step": current_step,
            "prompt": prompt,
            "predicted_answer": best_letter,
            "model_raw_response": last_response,
        }

    for attempt in range(1, max_attempts + 1):
        rate_limiter.acquire()

        timeout = 120.0 if attempt <= 2 else min(180.0 + 30.0 * (attempt - 3), 360.0)
        response_text = query_llm(prompt, api_key, model_name, timeout)
        last_response = response_text

        letter = extract_answer_letter(response_text, valid_letters)
        if letter:
            best_letter = letter
            break

        if response_text.startswith("[Error]"):
            preview = response_text[:120].replace("\n", " ")
            log(f"[WARN] id={sample_id} attempt {attempt} error: {preview}")
        elif attempt == max_attempts:
            log(f"[WARN] id={sample_id} 用盡 {max_attempts} 次仍無合法 letter；raw={response_text[:80]!r}")

        time.sleep(min(2.0 * attempt, 30.0))

    return {
        "id": sample_id,
        "current_step": current_step,
        "prompt": prompt,
        "predicted_answer": best_letter,
        "model_raw_response": last_response,
    }


def _build_submission_rows(samples: List[dict], results_by_id: Dict[int, dict]) -> List[dict]:
    rows = []
    for s in samples:
        sid = s["id"]
        letter = (results_by_id.get(sid) or {}).get("predicted_answer") or ""
        # 萬一仍空白，給個保底 "A"，避免 Kaggle 拒收
        if not letter:
            letter = sorted((s.get("options") or {}).keys())[:1] or ["A"]
            letter = letter[0]
        rows.append({"id": sid, "answer": letter})
    return rows


def run_pipeline(
    input_path: str,
    output_csv: str,
    backup_json: str,
    api_key_path: str,
    model_name: str,
    max_attempts: int,
    rate_per_minute: int,
    workers_per_key: int,
    save_every: int,
) -> None:
    api_keys = load_api_keys(api_key_path)
    samples = load_jsonl(input_path)
    log(f"載入 {len(samples)} 筆樣本：{input_path}")
    log(f"使用模型：{model_name}")

    existing = load_existing_results(backup_json)
    if existing:
        log(f"偵測到既有結果 {len(existing)} 筆：{backup_json}")

    results_by_id: Dict[int, dict] = dict(existing)
    results_lock = threading.Lock()
    save_lock = threading.Lock()

    to_process: List[dict] = []
    skipped = 0
    for sample in samples:
        sid = sample["id"]
        prev = results_by_id.get(sid)
        prev_letter = (prev or {}).get("predicted_answer", "") or ""
        valid_letters = set((sample.get("options") or {}).keys())
        if prev_letter in valid_letters:
            skipped += 1
        else:
            to_process.append(sample)

    n_keys = len(api_keys)
    total_workers = workers_per_key * n_keys
    log(f"沿用 {skipped} 筆；需要重打 {len(to_process)} 筆")
    log(f"偵測到 {n_keys} 把 API key → 啟用 {total_workers} 條 thread "
        f"({workers_per_key} workers × {n_keys} keys)")
    log(f"每把 key 各自 rate limit: {rate_per_minute} calls / 60s "
        f"(全域聚合上限 ≈ {rate_per_minute * n_keys} calls / 60s)")

    # 每把 key 各自一個 RateLimiter — NIM 的配額是 per-key 的，所以不該共用視窗
    rate_limiters = [
        RateLimiter(max_calls=rate_per_minute, window_seconds=60.0)
        for _ in api_keys
    ]

    completed = 0
    total = len(to_process)
    unresolved: List[int] = []

    def _worker(sample: dict, key_idx: int) -> str:
        sid = sample["id"]
        api_key = api_keys[key_idx]
        rate_limiter = rate_limiters[key_idx]
        with results_lock:
            prev = results_by_id.get(sid)
        try:
            result = process_sample(
                sample,
                api_key=api_key,
                rate_limiter=rate_limiter,
                model_name=model_name,
                max_attempts=max_attempts,
                previous_best=prev,
            )
        except Exception as exc:  # pragma: no cover
            log(f"[WARN] id={sid} 發生例外，保留舊值: {exc}")
            result = prev or {
                "id": sid,
                "current_step": sample.get("current_step", ""),
                "prompt": "",
                "predicted_answer": "",
                "model_raw_response": "",
            }
        with results_lock:
            results_by_id[sid] = result
        return result.get("predicted_answer", "") or ""

    if to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor, \
                tqdm(total=len(samples), initial=skipped, desc="處理", unit="q",
                     dynamic_ncols=True) as bar:
            bar.set_postfix(skip=skipped, miss=0)
            # round-robin 把每筆樣本分派給某一把 key
            future_to_sample = {
                executor.submit(_worker, sample, i % n_keys): sample
                for i, sample in enumerate(to_process)
            }
            for future in concurrent.futures.as_completed(future_to_sample):
                sample = future_to_sample[future]
                sid = sample["id"]
                try:
                    letter = future.result()
                except Exception as exc:  # pragma: no cover
                    log(f"[WARN] id={sid} worker 失敗: {exc}")
                    letter = ""

                if not letter:
                    unresolved.append(sid)

                completed += 1
                bar.set_postfix(skip=skipped, miss=len(unresolved))
                bar.update(1)

                if completed % save_every == 0 or completed == total:
                    with save_lock, results_lock:
                        rows = _build_submission_rows(samples, results_by_id)
                        try:
                            save_outputs(output_csv, backup_json, rows, results_by_id)
                        except Exception as exc:
                            log(f"[WARN] 寫檔失敗，跳過此次 flush，下次再試: {exc}")

    with save_lock, results_lock:
        rows = _build_submission_rows(samples, results_by_id)
        try:
            save_outputs(output_csv, backup_json, rows, results_by_id)
        except Exception as exc:
            log(f"[WARN] 最終寫檔失敗，請手動重跑: {exc}")

    log("\n" + "=" * 60)
    log(f"完成。樣本總數 {len(samples)}：沿用 {skipped} 筆 / 重打 {len(to_process)} 筆")
    log(f"輸出：{output_csv}")
    log(f"備份：{backup_json}")
    if unresolved:
        log(f"[WARN] 仍未取到合法 letter 的 id 共 {len(unresolved)} 筆 (CSV 已 fallback 為 A)："
            f"{unresolved[:30]}" + (" ..." if len(unresolved) > 30 else ""))
        log("       直接再執行一次本腳本即可從這些 id 繼續重試。")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW4 LLM Tool Calling Agent pipeline")
    parser.add_argument("--input", default=os.path.join("data", "test.jsonl"),
                        help="輸入的 JSONL 檔 (預設 data/test.jsonl)")
    parser.add_argument("--output-dir", default=None,
                        help="輸出資料夾 (預設 outputs/hw4/{model_slug}_{mmddhhmm})")
    parser.add_argument("--output", default="submission.csv",
                        help="輸出 CSV 檔名，會放進 --output-dir (預設 submission.csv)")
    parser.add_argument("--backup", default="llm_answering_results.json",
                        help="完整回應備份 JSON，會放進 --output-dir")
    parser.add_argument("--api-key", default="api_key.txt", help="API key 檔路徑 (預設 api_key.txt)")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"LLM model id (預設 {MODEL_NAME})")
    parser.add_argument("--max-attempts", type=int, default=10,
                        help="同一筆最多重試幾次 (預設 10)")
    parser.add_argument("--rate", type=int, default=37,
                        help="每把 API key 60 秒滑動視窗內最多呼叫次數 (預設 37)")
    parser.add_argument("--workers", type=int, default=4,
                        help="每把 API key 配的 worker 執行緒數 (預設 4)；"
                             "實際總 thread 數 = workers × api_key 數")
    parser.add_argument("--save-every", type=int, default=1,
                        help="每完成 N 筆就 flush 到磁碟 (預設 1)")
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()

    output_dir = args.output_dir or default_output_dir(args.model)
    os.makedirs(output_dir, exist_ok=True)

    output_csv = args.output if os.path.isabs(args.output) else os.path.join(output_dir, args.output)
    backup_json = args.backup if os.path.isabs(args.backup) else os.path.join(output_dir, args.backup)

    log(f"輸出資料夾: {output_dir}")

    run_pipeline(
        input_path=args.input,
        output_csv=output_csv,
        backup_json=backup_json,
        api_key_path=args.api_key,
        model_name=args.model,
        max_attempts=args.max_attempts,
        rate_per_minute=args.rate,
        workers_per_key=args.workers,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
