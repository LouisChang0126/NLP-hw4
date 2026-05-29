"""
HW4 Local Qwen3.6-27B + ambiguity-heavy 3-shot in-context inference.

- 後端：llama-cpp-python 載入 Qwen3.6-27B GGUF (預設 Q4_K_M)
- Prompt：concise + 3 個 ambiguity-heavy 範例 (train ids 3961/9642/3547)
- 範例設計：
    id=3961 (gold C, 4 工具全 `home_cleaning_*`，6 對歧義)
    id=9642 (gold E, 5 工具 `foreign_currency_*`，3 對歧義)
    id=3547 (gold C, train_ticket_query/cancelling/booking + search_train)
- 後處理：multi-tier regex 抽單一 letter
- GPU：單卡或雙卡 (雙卡時各載一份模型，thread pool 平行)

用法 (conda env: NLP2)：
  # 雙卡平行 (預設)
  python local_llm_inference.py \
      --gguf models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
      --devices 0,1

  # 單卡
  python local_llm_inference.py --gguf <path> --devices 0
"""

from __future__ import annotations

import argparse
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
from tqdm.auto import tqdm

Llama = None  # type: ignore


# ---------------------------------------------------------------------------
# Console / logging
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


def log(msg: str) -> None:
    try:
        tqdm.write(msg)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Answer parser — 多層 regex 回退
# ---------------------------------------------------------------------------

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
    last: str = ""
    for m in _STANDALONE_RE.finditer(response_text):
        letter = m.group(1).upper()
        if letter in valid_letters:
            last = letter
    return last


# ---------------------------------------------------------------------------
# Few-shot prompt builder
# ---------------------------------------------------------------------------

# 預設 ambiguity-heavy 3-shot (Kaggle LB 0.95197 ensemble 用的同樣 examples)
_FEWSHOT_IDS_DEFAULT: List[int] = [3961, 9642, 3547]

# 載入後 cache，避免每次重讀 train.jsonl
_EXAMPLES_CACHE: Optional[str] = None
_CACHED_IDS: Optional[List[int]] = None


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


def _format_one_example(sample: dict) -> str:
    options = sample.get("options", {}) or {}
    options_block = "\n".join(
        _format_tool(letter, options[letter]) for letter in sorted(options.keys())
    )
    valid_letters = ", ".join(sorted(options.keys()))
    return (
        f"Full context:\n{sample.get('full_context', '')}\n\n"
        f"Current step:\n{sample.get('current_step', '')}\n\n"
        f"Candidate tools:\n{options_block}\n\n"
        f"Valid option letters: {valid_letters}\n"
        f"Answer: {sample.get('answer', '')}"
    )


def _load_fewshot_examples(ids: List[int],
                           train_path: str = os.path.join("data", "train.jsonl")) -> List[dict]:
    wanted = set(ids)
    found: Dict[int, dict] = {}
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("id") in wanted:
                found[s["id"]] = s
                if len(found) == len(wanted):
                    break
    if len(found) != len(wanted):
        missing = wanted - set(found.keys())
        log(f"[FATAL] 找不到 few-shot 範例 ids: {missing}"); sys.exit(1)
    return [found[sid] for sid in ids]


def _get_examples_block(ids: List[int]) -> str:
    global _EXAMPLES_CACHE, _CACHED_IDS
    if _EXAMPLES_CACHE is not None and _CACHED_IDS == ids:
        return _EXAMPLES_CACHE
    samples = _load_fewshot_examples(ids)
    blocks = [_format_one_example(s) for s in samples]
    labeled = [f"[Example {i + 1}]\n{b}" for i, b in enumerate(blocks)]
    _EXAMPLES_CACHE = "\n\n".join(labeled)
    _CACHED_IDS = list(ids)
    log(f"[INFO] 已載入 {len(ids)} 個 few-shot 範例 (ids={ids})")
    return _EXAMPLES_CACHE


def build_prompt(full_context: str, current_step: str, options: Dict[str, dict],
                 fewshot_ids: List[int]) -> str:
    """Concise prompt + N-shot in-context examples."""
    options_block = "\n".join(
        _format_tool(letter, options[letter]) for letter in sorted(options.keys())
    )
    valid_letters = ", ".join(sorted(options.keys()))
    examples = _get_examples_block(fewshot_ids)
    return (
        "You are an expert tool-calling agent. Given the user's task context and the current step, "
        "select the correct tool from the candidate options.\n\n"
        f"Here are {len(fewshot_ids)} worked examples to demonstrate the format and reasoning:\n\n"
        f"{examples}\n\n"
        "[Now solve this one]\n"
        f"Full context:\n{full_context}\n\n"
        f"Current step:\n{current_step}\n\n"
        f"Candidate tools:\n{options_block}\n\n"
        f"Valid option letters: {valid_letters}\n"
        "Output ONLY a single uppercase letter from the valid options corresponding to the correct tool. "
        "No explanation, no analysis, no other text. (e.g., D)"
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        log(f"[FATAL] 找不到輸入檔 '{path}'"); sys.exit(1)
    samples: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log(f"[WARN] 第 {line_no} 行 JSON 解析失敗: {exc}")
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
            os.replace(src, dst); return
        except PermissionError as exc:
            last_exc = exc; time.sleep(delay); delay = min(delay * 2, 5.0)
    raise last_exc  # type: ignore[misc]


def save_outputs(output_csv: str, backup_json: str,
                 submission_rows: List[dict],
                 results_by_id: Dict[int, dict]) -> None:
    tmp_csv = output_csv + ".tmp"
    tmp_json = backup_json + ".tmp"
    pd.DataFrame(submission_rows).to_csv(tmp_csv, index=False)
    _atomic_replace(tmp_csv, output_csv)
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(list(results_by_id.values()), f, ensure_ascii=False, indent=2)
    _atomic_replace(tmp_json, backup_json)


def _build_submission_rows(samples: List[dict],
                           results_by_id: Dict[int, dict]) -> List[dict]:
    rows = []
    for s in samples:
        sid = s["id"]
        letter = (results_by_id.get(sid) or {}).get("predicted_answer") or ""
        if not letter:
            letter = sorted((s.get("options") or {}).keys())[:1] or ["A"]
            letter = letter[0]
        rows.append({"id": sid, "answer": letter})
    return rows


# ---------------------------------------------------------------------------
# Llama wrapper
# ---------------------------------------------------------------------------

def make_llama(gguf_path: str, device_idx: int, n_gpu_layers: int,
               n_ctx: int, n_threads: int, n_batch: int, seed: int):
    """載入一份 Llama 實例，整顆 pin 到 device_idx。"""
    global Llama
    if Llama is None:
        from llama_cpp import Llama as _Llama
        Llama = _Llama
    log(f"[INFO] 載入 GGUF 到 GPU {device_idx}: {gguf_path}")
    t0 = time.time()
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        seed=seed,
        verbose=False,
        main_gpu=device_idx,
        split_mode=0,  # LLAMA_SPLIT_MODE_NONE — 整顆放在 main_gpu
    )
    log(f"[INFO] GPU {device_idx} 模型載入完成，耗時 {time.time() - t0:.1f}s "
        f"(n_ctx={n_ctx})")
    return llm


class LlmWorker:
    """一張 GPU + 一份 Llama 實例 + 一把鎖。"""

    def __init__(self, llm, device_idx: int) -> None:
        self.llm = llm
        self.device_idx = device_idx
        self.lock = threading.Lock()


def _wrap_qwen3_no_think(user_msg: str) -> str:
    """Qwen3.6 ChatML + 空 <think></think> block (跳過 thinking)。"""
    return (
        "<|im_start|>user\n"
        + user_msg
        + "\n<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def query_local_llm(llm, prompt: str, max_tokens: int,
                    temperature: float, top_p: float) -> str:
    try:
        wrapped = _wrap_qwen3_no_think(prompt)
        resp = llm.create_completion(
            prompt=wrapped,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        choices = resp.get("choices") or []
        if not choices:
            return "[Error] no choices"
        text = choices[0].get("text") or ""
        return text.strip() if text else "[Error] empty content"
    except Exception as exc:
        return f"[Error] inference exception: {exc}"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def process_sample(llm, sample: dict, fewshot_ids: List[int],
                   max_attempts: int, max_tokens: int,
                   temperature: float, top_p: float,
                   previous_best: Optional[dict] = None) -> dict:
    sample_id = sample["id"]
    options = sample.get("options", {}) or {}
    valid_letters: Set[str] = set(options.keys())
    prompt = build_prompt(
        sample.get("full_context", ""),
        sample.get("current_step", ""),
        options, fewshot_ids,
    )

    best_letter = (previous_best or {}).get("predicted_answer", "") or ""
    last_response = (previous_best or {}).get("model_raw_response", "")

    if best_letter in valid_letters:
        return {
            "id": sample_id,
            "current_step": sample.get("current_step", ""),
            "predicted_answer": best_letter,
            "model_raw_response": last_response,
        }

    for attempt in range(1, max_attempts + 1):
        t = temperature if attempt == 1 else min(temperature + 0.05 * attempt, 0.1)
        response_text = query_local_llm(llm, prompt, max_tokens, t, top_p)
        last_response = response_text
        letter = extract_answer_letter(response_text, valid_letters)
        if letter:
            best_letter = letter
            break
        if response_text.startswith("[Error]"):
            log(f"[WARN] id={sample_id} attempt {attempt} error: "
                f"{response_text[:120].replace(chr(10), ' ')}")
        elif attempt == max_attempts:
            log(f"[WARN] id={sample_id} 用盡 {max_attempts} 次仍無合法 letter；"
                f"raw={response_text[:80]!r}")

    return {
        "id": sample_id,
        "current_step": sample.get("current_step", ""),
        "predicted_answer": best_letter,
        "model_raw_response": last_response,
    }


def run_pipeline(input_path: str, output_csv: str, backup_json: str,
                 gguf_path: str, devices: List[int], fewshot_ids: List[int],
                 n_gpu_layers: int, n_ctx: int, n_threads: int, n_batch: int,
                 seed: int, max_attempts: int, max_tokens: int,
                 temperature: float, top_p: float, save_every: int) -> None:
    samples = load_jsonl(input_path)
    log(f"載入 {len(samples)} 筆樣本：{input_path}")
    # 預先載入 examples
    _ = _get_examples_block(fewshot_ids)

    existing = load_existing_results(backup_json)
    if existing:
        log(f"偵測到既有結果 {len(existing)} 筆：{backup_json}")
    results_by_id: Dict[int, dict] = dict(existing)

    to_process: List[dict] = []
    skipped = 0
    for sample in samples:
        sid = sample["id"]
        prev_letter = (results_by_id.get(sid) or {}).get("predicted_answer", "") or ""
        valid_letters = set((sample.get("options") or {}).keys())
        if prev_letter in valid_letters:
            skipped += 1
        else:
            to_process.append(sample)

    log(f"沿用 {skipped} 筆；需要重打 {len(to_process)} 筆")
    if not to_process:
        rows = _build_submission_rows(samples, results_by_id)
        save_outputs(output_csv, backup_json, rows, results_by_id)
        return

    log(f"準備在 {len(devices)} 張 GPU 上各載入一份模型: {devices}")
    workers: List[LlmWorker] = []
    for dev in devices:
        llm = make_llama(
            gguf_path=gguf_path, device_idx=dev,
            n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
            n_threads=n_threads, n_batch=n_batch, seed=seed,
        )
        workers.append(LlmWorker(llm, dev))
    n_workers = len(workers)
    log(f"[INFO] 共 {n_workers} 條 worker thread")

    results_lock = threading.Lock()
    save_lock = threading.Lock()
    completed = 0
    total = len(to_process)

    def _worker_fn(sample: dict, worker_idx: int) -> str:
        sid = sample["id"]
        w = workers[worker_idx]
        with results_lock:
            prev = results_by_id.get(sid)
        try:
            with w.lock:
                result = process_sample(
                    w.llm, sample, fewshot_ids,
                    max_attempts=max_attempts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    previous_best=prev,
                )
        except Exception as exc:  # pragma: no cover
            log(f"[WARN] id={sid} (GPU {w.device_idx}) 例外: {exc}")
            result = prev or {
                "id": sid,
                "current_step": sample.get("current_step", ""),
                "predicted_answer": "", "model_raw_response": "",
            }
        with results_lock:
            results_by_id[sid] = result
        return result.get("predicted_answer", "") or ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex, \
            tqdm(total=len(samples), initial=skipped, desc="處理", unit="q",
                 dynamic_ncols=True) as bar:
        future_to_sample = {
            ex.submit(_worker_fn, sample, i % n_workers): sample
            for i, sample in enumerate(to_process)
        }
        for future in concurrent.futures.as_completed(future_to_sample):
            completed += 1
            try:
                future.result()
            except Exception as exc:
                log(f"[WARN] worker fail: {exc}")
            bar.update(1)
            if completed % save_every == 0 or completed == total:
                with save_lock, results_lock:
                    rows = _build_submission_rows(samples, results_by_id)
                    try:
                        save_outputs(output_csv, backup_json, rows, results_by_id)
                    except Exception as exc:
                        log(f"[WARN] flush 失敗: {exc}")

    with save_lock, results_lock:
        rows = _build_submission_rows(samples, results_by_id)
        save_outputs(output_csv, backup_json, rows, results_by_id)
    log(f"完成 → {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_devices(s: str) -> List[int]:
    parts = [p for p in re.split(r"[,;\s]+", s.strip()) if p]
    try:
        devices = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--devices 必須是整數或以逗號分隔 (got {s!r})") from exc
    if not devices or len(devices) > 2 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError(f"--devices 必須是 1 或 2 個唯一整數 (got {devices})")
    return devices


def _parse_fewshot_ids(s: str) -> List[int]:
    return [int(x) for x in re.split(r"[,;\s]+", s.strip()) if x]


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(name)).strip("_") or "model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HW4 local Qwen3.6 + ambiguity-heavy 3-shot inference")
    p.add_argument("--input", default=os.path.join("data", "test.jsonl"))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--backup", default="llm_answering_results.json")
    p.add_argument("--gguf", required=True, help="GGUF 檔路徑")
    p.add_argument("--devices", type=_parse_devices, default=[0, 1],
                   help="GPU index (1 或 2 張)，預設 '0,1'")
    p.add_argument("--fewshot-ids", type=_parse_fewshot_ids,
                   default=_FEWSHOT_IDS_DEFAULT,
                   help=f"逗號分隔的 train.jsonl id (預設 ambig: {_FEWSHOT_IDS_DEFAULT})")
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--n-ctx", type=int, default=8192,
                   help="3-shot prompt ~5000 tokens；預設 8192")
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--n-batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.1)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--save-every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        "outputs",
        f"hw4_local_{slugify(args.gguf)}_"
        f"{_dt.datetime.now().strftime('%m%d%H%M')}",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, args.output)
    backup_json = os.path.join(output_dir, args.backup)
    log(f"輸出資料夾: {output_dir}")

    run_pipeline(
        input_path=args.input,
        output_csv=output_csv,
        backup_json=backup_json,
        gguf_path=args.gguf,
        devices=args.devices,
        fewshot_ids=args.fewshot_ids,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        seed=args.seed,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
