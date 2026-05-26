"""
HW4 dev-set 評估腳本：對應 spec Q2 (Prompt × Full-Info vs Prompt × Structural-Only)
+ Q3 (tool ambiguity 錯誤分析)。

設計理念
========
基於 local_llm_inference.py，加入：
  1. dev split：從 train.jsonl 取最後 N 筆當 dev (有 gold answer 可算 accuracy)
  2. --prompt-mode {full, struct}：
       - full   = 完整工具資訊 (name + description + 參數 keys/types/descriptions
                  + 回傳 keys/types/descriptions)
       - struct = 只保留參數 keys + types。spec Q2 規定的 Structural-Only 設定：
                  完全移除 tool name、tool description、argument descriptions、
                  result descriptions
  3. 推論完自動做：
       - accuracy / per-letter 統計 / confusion matrix
       - Q3 錯誤分類：把錯題分成 ambiguity error (predicted tool 與 gold tool
         的 name token Jaccard ≥ threshold) vs other error

GPU 配置：預設 --devices 0,1，兩張卡各載一份模型，thread pool 平行分派 dev 樣本。

用法 (conda env: NLP2)：
  # Full-Info prompt 在 dev 上跑
  python run_dev_eval.py --gguf <path> --devices 0,1 --prompt-mode full \
      --output-dir outputs/exp_dev_prompt_full

  # Structural-Only prompt 在 dev 上跑
  python run_dev_eval.py --gguf <path> --devices 0,1 --prompt-mode struct \
      --output-dir outputs/exp_dev_prompt_struct
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
# Answer parser — 對齊 local_llm_inference.py
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
# Prompt builders — Full vs Structural-Only
# ---------------------------------------------------------------------------

def _format_schema_full(schema: dict) -> str:
    """完整 schema 格式：key (type): description"""
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


def _format_schema_struct(schema: dict) -> str:
    """Structural-Only schema：只剩 key + type，描述完全剔除"""
    props = (schema or {}).get("properties") or {}
    if not props:
        return "  (none)"
    return "\n".join(
        f"  - {key} ({(spec or {}).get('type', 'any')})"
        for key, spec in props.items()
    )


def _format_tool_full(letter: str, tool: dict) -> str:
    name = tool.get("name", "")
    desc = tool.get("description", "")
    args = tool.get("arguments") or {}
    results = tool.get("results") or {}
    parts = [
        f"[{letter}] name: {name}",
        f"    description: {desc}" if desc else "",
        "    arguments:",
        _format_schema_full(args),
        "    results:",
        _format_schema_full(results),
    ]
    return "\n".join(p for p in parts if p != "")


def _format_tool_struct(letter: str, tool: dict) -> str:
    """Structural-Only：spec 規定徹底剔除 tool name / description / 各層描述，
    只保留參數 keys + types 跟 回傳 keys + types。"""
    args = tool.get("arguments") or {}
    results = tool.get("results") or {}
    parts = [
        f"[{letter}]",
        "    arguments:",
        _format_schema_struct(args),
        "    results:",
        _format_schema_struct(results),
    ]
    return "\n".join(parts)


_PROMPT_HEADER_FULL = (
    "You are an expert tool-calling agent. Given the historical context of a user's task, "
    "the current sub-step to be executed, and a list of candidate tools, your job is to "
    "select the SINGLE correct tool whose name, arguments, and returned results best "
    "match what the current step needs to do.\n\n"
    "Reasoning guidelines:\n"
    "  - Match the tool's *action* (its name + description) to the current step's verb "
    "(query / search / book / confirm / etc.).\n"
    "  - Two tools may have similar names (e.g. `search_train` vs `query_past_ticket`); "
    "disambiguate by carefully comparing each tool's argument keys and result keys against "
    "what the current step actually requires.\n"
    "  - Prefer the tool whose required inputs are already available in the full_context.\n"
    "  - The current_step's wording is the strongest signal — only one option will fit exactly.\n\n"
)

_PROMPT_HEADER_STRUCT = (
    "You are an expert tool-calling agent. Given the historical context of a user's task, "
    "the current sub-step to be executed, and a list of candidate tools — but in this setting "
    "**tool names and all natural-language descriptions have been stripped**. You only see each "
    "tool's argument keys + types and result keys + types. Pick the SINGLE tool whose schema "
    "structure best fits the current step.\n\n"
    "Reasoning guidelines:\n"
    "  - Parameter key NAMES (e.g. `departure_time`, `ticket_id`, `seat_type`) themselves carry "
    "strong hints about what the tool does — read them carefully.\n"
    "  - Match the set of required inputs (current_step + full_context) to the tool whose "
    "argument keys cover those inputs.\n"
    "  - Match the kind of output the step expects to the tool whose result keys produce it.\n"
    "  - When two tools share many argument keys, look at the result keys to disambiguate.\n\n"
)


def build_prompt(full_context: str, current_step: str, options: Dict[str, dict],
                 prompt_mode: str) -> str:
    if prompt_mode == "full":
        formatter = _format_tool_full
        header = _PROMPT_HEADER_FULL
    elif prompt_mode == "struct":
        formatter = _format_tool_struct
        header = _PROMPT_HEADER_STRUCT
    else:
        raise ValueError(f"prompt_mode 必須是 'full' 或 'struct'，got {prompt_mode!r}")

    options_block = "\n".join(
        formatter(letter, options[letter]) for letter in sorted(options.keys())
    )
    valid_letters = ", ".join(sorted(options.keys()))
    return (
        header
        + f"Full context (the user's overall plan so far):\n{full_context}\n\n"
        + f"Current step (the sub-task to execute right now):\n{current_step}\n\n"
        + f"Candidate tools:\n{options_block}\n\n"
        + f"Valid option letters: {valid_letters}\n"
        + "Output ONLY the final answer on its own line in this exact format:\n"
          "Answer: <LETTER>\n"
          "where <LETTER> is one of the valid option letters above. "
          "Do not output any explanation, analysis, or anything after that line."
    )


# ---------------------------------------------------------------------------
# Persistence — 對齊 local_llm_inference.py
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
    submission_csv: str,
    backup_json: str,
    submission_rows: List[dict],
    results_by_id: Dict[int, dict],
) -> None:
    tmp_csv = submission_csv + ".tmp"
    tmp_json = backup_json + ".tmp"
    pd.DataFrame(submission_rows).to_csv(tmp_csv, index=False)
    _atomic_replace(tmp_csv, submission_csv)
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(list(results_by_id.values()), f, ensure_ascii=False, indent=2)
    _atomic_replace(tmp_json, backup_json)


def _build_submission_rows(samples: List[dict], results_by_id: Dict[int, dict]) -> List[dict]:
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
# Dev split
# ---------------------------------------------------------------------------

def build_or_load_dev_split(train_path: str, dev_path: str, dev_size: int) -> List[dict]:
    """從 train.jsonl 取最後 dev_size 筆當 dev split (deterministic)。
    若 dev_path 已存在則直接沿用，確保 Full 和 Struct 兩次跑用同一份 dev。"""
    if os.path.exists(dev_path):
        existing = load_jsonl(dev_path)
        log(f"[INFO] 沿用既有 dev split: {dev_path} ({len(existing)} 筆)")
        return existing
    log(f"[INFO] 從 {train_path} 取最後 {dev_size} 筆當 dev split")
    train_samples = load_jsonl(train_path)
    if len(train_samples) < dev_size:
        log(f"[WARN] train.jsonl 只有 {len(train_samples)} 筆 (< dev_size={dev_size})，全部當 dev")
        dev = train_samples
    else:
        dev = train_samples[-dev_size:]
    os.makedirs(os.path.dirname(dev_path) or ".", exist_ok=True)
    with open(dev_path, "w", encoding="utf-8") as f:
        for s in dev:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log(f"[INFO] 寫出 dev split → {dev_path}")
    return dev


# ---------------------------------------------------------------------------
# Llama wrapper — 對齊 local_llm_inference.py，每張卡一份模型
# ---------------------------------------------------------------------------

def make_llama(
    gguf_path: str,
    device_idx: int,
    n_gpu_layers: int,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    seed: int,
):
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
        split_mode=0,  # LLAMA_SPLIT_MODE_NONE — 整顆 pin 在 main_gpu
    )
    log(f"[INFO] GPU {device_idx} 模型載入完成，耗時 {time.time() - t0:.1f} s")
    return llm


class LlmWorker:
    def __init__(self, llm, device_idx: int) -> None:
        self.llm = llm
        self.device_idx = device_idx
        self.lock = threading.Lock()


def _wrap_qwen3_no_think(user_msg: str) -> str:
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
# Core inference loop
# ---------------------------------------------------------------------------

def process_sample(
    llm,
    sample: dict,
    prompt_mode: str,
    max_attempts: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    previous_best: Optional[dict] = None,
) -> dict:
    sample_id = sample["id"]
    full_context = sample.get("full_context", "")
    current_step = sample.get("current_step", "")
    options = sample.get("options", {}) or {}
    valid_letters: Set[str] = set(options.keys())
    prompt = build_prompt(full_context, current_step, options, prompt_mode)

    best_letter = (previous_best or {}).get("predicted_answer", "") or ""
    last_response = (previous_best or {}).get("model_raw_response", "")

    if best_letter in valid_letters:
        return {
            "id": sample_id,
            "current_step": current_step,
            "prompt_mode": prompt_mode,
            "predicted_answer": best_letter,
            "model_raw_response": last_response,
        }

    for attempt in range(1, max_attempts + 1):
        t = temperature if attempt == 1 else min(temperature + 0.1 * attempt, 0.9)
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
        "current_step": current_step,
        "prompt_mode": prompt_mode,
        "predicted_answer": best_letter,
        "model_raw_response": last_response,
    }


# ---------------------------------------------------------------------------
# Evaluation + Q3 error analysis
# ---------------------------------------------------------------------------

def _tokenize_tool_name(name: str) -> Set[str]:
    """把 snake_case / kebab-case tool name 拆成 token set (小寫)。"""
    if not name:
        return set()
    return {t for t in re.split(r"[_\-\s]+", name.lower()) if t}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def evaluate_and_analyze(
    results_by_id: Dict[int, dict],
    dev_samples: List[dict],
    ambiguity_threshold: float,
) -> dict:
    """算 accuracy + Q3 ambiguity error 比例。

    ambiguity error 判準：predicted tool name 與 gold tool name 的
    snake_case token Jaccard ≥ threshold (預設 0.4)。
    例：train_ticket_query vs search_train 共享 train → Jaccard=1/4=0.25 (非)，
        train_ticket_query vs train_ticket_search → 共享 {train, ticket} → 2/4=0.5 (是)。
    """
    total = 0
    correct = 0
    per_gold = collections.Counter()       # gold letter → count
    per_gold_correct = collections.Counter()
    confusion = collections.Counter()      # (gold, pred) → count
    errors: List[dict] = []

    for sample in dev_samples:
        sid = sample["id"]
        gold = sample.get("answer", "")
        if not gold:
            continue
        pred = (results_by_id.get(sid) or {}).get("predicted_answer", "") or ""
        if not pred:
            continue
        total += 1
        per_gold[gold] += 1
        if pred == gold:
            correct += 1
            per_gold_correct[gold] += 1
            continue
        confusion[(gold, pred)] += 1

        options = sample.get("options", {}) or {}
        gold_name = (options.get(gold) or {}).get("name", "")
        pred_name = (options.get(pred) or {}).get("name", "")
        gold_toks = _tokenize_tool_name(gold_name)
        pred_toks = _tokenize_tool_name(pred_name)
        jacc = _jaccard(gold_toks, pred_toks)
        is_amb = jacc >= ambiguity_threshold

        errors.append({
            "id": sid,
            "gold": gold,
            "pred": pred,
            "gold_tool": gold_name,
            "pred_tool": pred_name,
            "jaccard": round(jacc, 3),
            "ambiguity": is_amb,
            "current_step": sample.get("current_step", ""),
        })

    accuracy = correct / total if total else 0.0
    n_errors = len(errors)
    n_amb = sum(1 for e in errors if e["ambiguity"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "errors": n_errors,
        "ambiguity_errors": n_amb,
        "ambiguity_pct_of_errors": (n_amb / n_errors) if n_errors else 0.0,
        "ambiguity_pct_of_total": (n_amb / total) if total else 0.0,
        "per_letter": {
            L: {"total": int(per_gold[L]), "correct": int(per_gold_correct[L]),
                "acc": (per_gold_correct[L] / per_gold[L]) if per_gold[L] else 0.0}
            for L in sorted(per_gold)
        },
        "confusion_top10": collections.Counter(
            {f"{g}->{p}": c for (g, p), c in confusion.items()}
        ).most_common(10),
        "error_details": errors,
        "ambiguity_threshold": ambiguity_threshold,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dev_samples: List[dict],
    submission_csv: str,
    backup_json: str,
    eval_json: str,
    gguf_path: str,
    devices: List[int],
    n_gpu_layers: int,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    seed: int,
    prompt_mode: str,
    max_attempts: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    save_every: int,
    ambiguity_threshold: float,
) -> None:
    log(f"dev 樣本數: {len(dev_samples)}；prompt_mode={prompt_mode}")

    existing = load_existing_results(backup_json)
    if existing:
        log(f"偵測到既有結果 {len(existing)} 筆：{backup_json}")
    results_by_id: Dict[int, dict] = dict(existing)

    # 沿用條件：先前的結果是同樣 prompt_mode 且 letter 合法
    to_process: List[dict] = []
    skipped = 0
    for sample in dev_samples:
        sid = sample["id"]
        prev = results_by_id.get(sid)
        prev_letter = (prev or {}).get("predicted_answer", "") or ""
        prev_mode = (prev or {}).get("prompt_mode", "")
        valid_letters = set((sample.get("options") or {}).keys())
        if prev_letter in valid_letters and prev_mode == prompt_mode:
            skipped += 1
        else:
            to_process.append(sample)
    log(f"沿用 {skipped} 筆；需要重打 {len(to_process)} 筆")

    if to_process:
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
                        w.llm, sample,
                        prompt_mode=prompt_mode,
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
                    "prompt_mode": prompt_mode,
                    "predicted_answer": "",
                    "model_raw_response": "",
                }
            with results_lock:
                results_by_id[sid] = result
            return result.get("predicted_answer", "") or ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex, \
                tqdm(total=len(dev_samples), initial=skipped, desc=f"處理({prompt_mode})",
                     unit="q", dynamic_ncols=True) as bar:
            future_to_sample = {
                ex.submit(_worker_fn, sample, i % n_workers): sample
                for i, sample in enumerate(to_process)
            }
            for future in concurrent.futures.as_completed(future_to_sample):
                completed += 1
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover
                    log(f"[WARN] worker 失敗: {exc}")
                bar.update(1)
                if completed % save_every == 0 or completed == total:
                    with save_lock, results_lock:
                        rows = _build_submission_rows(dev_samples, results_by_id)
                        try:
                            save_outputs(submission_csv, backup_json, rows, results_by_id)
                        except Exception as exc:
                            log(f"[WARN] flush 失敗: {exc}")

        with save_lock, results_lock:
            rows = _build_submission_rows(dev_samples, results_by_id)
            try:
                save_outputs(submission_csv, backup_json, rows, results_by_id)
            except Exception as exc:
                log(f"[WARN] 最終寫檔失敗: {exc}")

    # ── 評估 ──
    log("\n" + "=" * 60)
    log("評估中...")
    stats = evaluate_and_analyze(results_by_id, dev_samples, ambiguity_threshold)
    log(f"prompt_mode = {prompt_mode}")
    log(f"  dev total      = {stats['total']}")
    log(f"  correct        = {stats['correct']}")
    log(f"  accuracy       = {stats['accuracy']:.4f}")
    log(f"  total errors   = {stats['errors']}")
    log(f"  ambiguity err  = {stats['ambiguity_errors']}  "
        f"({stats['ambiguity_pct_of_errors']*100:.1f}% of errors, "
        f"{stats['ambiguity_pct_of_total']*100:.1f}% of dev)")
    log(f"  per-letter accuracy:")
    for L, info in stats["per_letter"].items():
        log(f"    {L}: {info['correct']}/{info['total']} = {info['acc']:.3f}")
    log(f"  top confusions (gold→pred): {stats['confusion_top10'][:5]}")

    # 寫 eval JSON
    stats_with_meta = dict(stats)
    stats_with_meta["prompt_mode"] = prompt_mode
    stats_with_meta["gguf"] = os.path.basename(gguf_path)
    stats_with_meta["devices"] = devices
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(stats_with_meta, f, ensure_ascii=False, indent=2)
    log(f"  寫出評估結果: {eval_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_devices(s: str) -> List[int]:
    parts = [p for p in re.split(r"[,;\s]+", s.strip()) if p]
    try:
        devices = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--devices 必須是整數或以逗號分隔的整數 (got {s!r})") from exc
    if not devices:
        raise argparse.ArgumentTypeError("--devices 不能為空")
    if len(devices) > 2:
        raise argparse.ArgumentTypeError(
            f"本腳本只支援單卡或雙卡 (got {devices})")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError(f"--devices 重複 ({devices})")
    return devices


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(name)).strip("_") or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HW4 dev-set evaluation: Prompt × {Full, Struct} + Q3 error analysis"
    )
    # I/O
    parser.add_argument("--train-input", default=os.path.join("data", "train.jsonl"),
                        help="train.jsonl 路徑 (用來抽 dev split)")
    parser.add_argument("--dev-output", default=os.path.join("data", "dev.jsonl"),
                        help="dev split 寫出位置；存在即沿用")
    parser.add_argument("--dev-size", type=int, default=1000,
                        help="從 train.jsonl 取最後幾筆當 dev (預設 1000)")
    parser.add_argument("--output-dir", default=None,
                        help="輸出資料夾 (預設 outputs/exp_dev_{prompt_mode}_{mmddhhmm})")
    # Model
    parser.add_argument("--gguf", required=True, help="GGUF 檔路徑")
    parser.add_argument("--devices", type=_parse_devices, default=[0, 1],
                        help="GPU index，例如 '0' / '1' / '0,1' (預設 '0,1' 雙卡)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    # Prompt + decoding
    parser.add_argument("--prompt-mode", required=True, choices=["full", "struct"],
                        help="full = 完整工具描述；struct = 只保留 keys+types (spec Q2 兩種設定)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=25)
    # Error analysis
    parser.add_argument("--ambiguity-threshold", type=float, default=0.4,
                        help="tool name Jaccard ≥ 此值即視為 ambiguity error (預設 0.4)")
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()

    output_dir = args.output_dir or os.path.join(
        "outputs",
        f"exp_dev_{args.prompt_mode}_{slugify(args.gguf)}_"
        f"{_dt.datetime.now().strftime('%m%d%H%M')}",
    )
    os.makedirs(output_dir, exist_ok=True)
    submission_csv = os.path.join(output_dir, "dev_submission.csv")
    backup_json = os.path.join(output_dir, "llm_answering_results.json")
    eval_json = os.path.join(output_dir, "eval_results.json")
    log(f"輸出資料夾: {output_dir}")

    dev_samples = build_or_load_dev_split(args.train_input, args.dev_output, args.dev_size)

    run_pipeline(
        dev_samples=dev_samples,
        submission_csv=submission_csv,
        backup_json=backup_json,
        eval_json=eval_json,
        gguf_path=args.gguf,
        devices=args.devices,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        seed=args.seed,
        prompt_mode=args.prompt_mode,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        save_every=args.save_every,
        ambiguity_threshold=args.ambiguity_threshold,
    )


if __name__ == "__main__":
    main()
