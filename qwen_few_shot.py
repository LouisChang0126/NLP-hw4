"""
HW4 LLM Tool Calling Agent — 本地 GGUF 推論版 + 3-shot in-context examples。

基於 HW4_111550132.py，差異：
  - build_prompt 在 user message 前面插入 3 個 train.jsonl 範例 (固定 id 16/17/582)
  - 三個範例涵蓋三個 gold letter (A/B/D) 與三個 domain (banking/flight/payment)
  - n_ctx 預設拉到 8192 (3-shot prompt 約 ~4500-5500 tokens)
  - 其餘 (concise 指令、empty <think>, parser、dual-GPU) 沿用 HW4_111550132.py

用法 (conda env: NLP2)：
  python qwen_few_shot.py --gguf models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
      --devices 0,1 --temperature 0.0
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

# 重用 HW4_111550132.py 的 building blocks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from HW4_111550132 import (  # noqa: E402
    _configure_stdout,
    log,
    extract_answer_letter,
    _format_tool,
    load_jsonl,
    load_existing_results,
    save_outputs,
    _build_submission_rows,
    slugify_model_name,
    make_llama,
    LlmWorker,
    _wrap_qwen3_no_think,
    query_local_llm,
    _parse_devices,
)


# ---------------------------------------------------------------------------
# Few-shot examples: 3 個固定樣本 (gold letter 多樣 + 跨 domain)
# ---------------------------------------------------------------------------

# 預設 3-shot 範例 id；CLI 可以用 --fewshot-ids 覆寫成任意 N-shot
_FEWSHOT_IDS: List[int] = [16, 17, 582]


def _format_one_example(sample: dict) -> str:
    """把一個 train 樣本格式化成 in-context example 區塊。"""
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


def _load_fewshot_examples(train_path: str = os.path.join("data", "train.jsonl")) -> List[str]:
    """從 train.jsonl 抓 _FEWSHOT_IDS 對應的樣本，回傳格式化好的 example block list。"""
    wanted = set(_FEWSHOT_IDS)
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
        log(f"[FATAL] 找不到 few-shot 範例 ids: {missing}")
        sys.exit(1)
    # 依 _FEWSHOT_IDS 的順序排列
    return [_format_one_example(found[sid]) for sid in _FEWSHOT_IDS]


_EXAMPLES_CACHE: Optional[str] = None


def _get_examples_block() -> str:
    global _EXAMPLES_CACHE
    if _EXAMPLES_CACHE is None:
        blocks = _load_fewshot_examples()
        labeled = [
            f"[Example {i + 1}]\n{b}"
            for i, b in enumerate(blocks)
        ]
        _EXAMPLES_CACHE = "\n\n".join(labeled)
        log(f"[INFO] 已載入 {len(blocks)} 個 few-shot 範例 (ids={_FEWSHOT_IDS})")
    return _EXAMPLES_CACHE


def build_prompt(full_context: str, current_step: str, options: Dict[str, dict]) -> str:
    """3-shot concise prompt：先給 3 個 worked examples，再問當前題目。"""
    options_block = "\n".join(
        _format_tool(letter, options[letter]) for letter in sorted(options.keys())
    )
    valid_letters = ", ".join(sorted(options.keys()))
    examples = _get_examples_block()
    return (
        "You are an expert tool-calling agent. Given the user's task context and the current step, "
        "select the correct tool from the candidate options.\n\n"
        "Here are 3 worked examples to demonstrate the format and reasoning:\n\n"
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
# Core loop — 與 HW4_111550132.py 結構一致，但用上面新的 build_prompt
# ---------------------------------------------------------------------------

def process_sample(
    llm,
    sample: dict,
    max_attempts: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
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
            "prompt_mode": "fewshot3",
            "predicted_answer": best_letter,
            "model_raw_response": last_response,
        }

    for attempt in range(1, max_attempts + 1):
        t = temperature if attempt == 1 else min(temperature + 0.05 * attempt, 0.1)
        response_text = query_local_llm(
            llm, prompt, max_tokens, t, top_p, disable_thinking=disable_thinking,
        )
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
        "current_step": current_step,
        "prompt_mode": "fewshot3",
        "predicted_answer": best_letter,
        "model_raw_response": last_response,
    }


def run_pipeline(
    input_path: str,
    output_csv: str,
    backup_json: str,
    gguf_path: str,
    devices: List[int],
    n_gpu_layers: int,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    seed: int,
    max_attempts: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
    save_every: int,
) -> None:
    samples = load_jsonl(input_path)
    log(f"載入 {len(samples)} 筆樣本：{input_path}")
    # 預先載入 examples block，順便檢查 token-ish 大小
    _ = _get_examples_block()

    existing = load_existing_results(backup_json)
    if existing:
        log(f"偵測到既有結果 {len(existing)} 筆：{backup_json}")
    results_by_id: Dict[int, dict] = dict(existing)

    to_process: List[dict] = []
    skipped = 0
    for sample in samples:
        sid = sample["id"]
        prev_letter = (results_by_id.get(sid) or {}).get("predicted_answer", "") or ""
        prev_mode = (results_by_id.get(sid) or {}).get("prompt_mode", "")
        valid_letters = set((sample.get("options") or {}).keys())
        if prev_letter in valid_letters and prev_mode == "fewshot3":
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
            n_threads=n_threads, n_batch=n_batch, seed=seed, chat_format=None,
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
                    w.llm, sample,
                    max_attempts=max_attempts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    disable_thinking=disable_thinking,
                    previous_best=prev,
                )
        except Exception as exc:  # pragma: no cover
            log(f"[WARN] id={sid} (GPU {w.device_idx}) 例外: {exc}")
            result = prev or {
                "id": sid,
                "current_step": sample.get("current_step", ""),
                "prompt_mode": "fewshot3",
                "predicted_answer": "",
                "model_raw_response": "",
            }
        with results_lock:
            results_by_id[sid] = result
        return result.get("predicted_answer", "") or ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex, \
            tqdm(total=len(samples), initial=skipped, desc="fewshot3", unit="q",
                 dynamic_ncols=True) as bar:
        future_to_sample = {
            ex.submit(_worker_fn, sample, i % n_workers): sample
            for i, sample in enumerate(to_process)
        }
        for future in concurrent.futures.as_completed(future_to_sample):
            completed += 1
            try:
                future.result()
            except Exception as exc:  # pragma: no cover
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
    log(f"\n完成。輸出: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW4 Qwen3.6 3-shot in-context inference")
    parser.add_argument("--input", default=os.path.join("data", "test.jsonl"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--backup", default="llm_answering_results.json")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--devices", type=_parse_devices, default=[0, 1])
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=8192,
                        help="3-shot prompt 約 4500-5500 tokens，預設 8192 留足空間")
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--fewshot-ids", default=",".join(str(x) for x in _FEWSHOT_IDS),
                        help="逗號分隔的 train.jsonl id (例如 '16,17,582,3961,9642')")
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()

    global _FEWSHOT_IDS, _EXAMPLES_CACHE
    parsed_ids = [int(x) for x in re.split(r"[,;\s]+", args.fewshot_ids.strip()) if x]
    if parsed_ids != _FEWSHOT_IDS:
        _FEWSHOT_IDS = parsed_ids
        _EXAMPLES_CACHE = None  # 強制重新載入

    output_dir = args.output_dir or os.path.join(
        "outputs", f"hw4_local_fewshot{len(_FEWSHOT_IDS)}_{slugify_model_name(args.gguf)}_"
                   f"{_dt.datetime.now().strftime('%m%d%H%M')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_csv = (args.output if os.path.isabs(args.output)
                  else os.path.join(output_dir, args.output))
    backup_json = (args.backup if os.path.isabs(args.backup)
                   else os.path.join(output_dir, args.backup))
    log(f"輸出資料夾: {output_dir}")

    run_pipeline(
        input_path=args.input,
        output_csv=output_csv,
        backup_json=backup_json,
        gguf_path=args.gguf,
        devices=args.devices,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        seed=args.seed,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        disable_thinking=True,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
