"""
HW4 LLM Tool Calling Agent — 本地 GGUF 推論版 (llama-cpp-python)。

設計目標 (對齊 HW4_111550132.py)：
  1. 讀 test.jsonl (id / full_context / current_step / options)
  2. 對每筆呼叫本地載入的 GGUF 模型挑出 (A..H)
  3. 自動重試直到拿到合法 letter，或耗盡 max_attempts
  4. 寫出 submission.csv + llm_answering_results.json
  5. 每筆即時 flush，可斷點續跑

差異：
  - 不打 API、直接本地推論；不需 api_key.txt
  - 單卡 / 雙卡 (本腳本只支援 1 或 2 張)：
      * 單卡：模型整顆裝在那張，sequential 推論
      * 雙卡：每張卡各載入一份完整模型，thread pool 平行分派樣本
        → 接近 2× 加速 (兩張卡沒有跨卡通訊)

用法 (conda env: NLP2)：
  # 單卡 (預設 GPU 0)：
  python local_llm_inference.py \
      --gguf models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
      --devices 0

  # 雙卡平行：
  python local_llm_inference.py \
      --gguf models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
      --devices 0,1
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

# llama_cpp 進來很慢 (CUDA init)，等真正要用時再 import
Llama = None  # type: ignore


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


def log(msg: str) -> None:
    try:
        tqdm.write(msg)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Prompt / parsing — 與 HW4_111550132.py 對齊
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


def slugify_model_name(name: str) -> str:
    tail = os.path.basename(name)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", tail).strip("_")
    return slug or "model"


def default_output_dir(model_path: str) -> str:
    stamp = _dt.datetime.now().strftime("%m%d%H%M")
    return os.path.join("outputs", "hw4_local", f"{slugify_model_name(model_path)}_{stamp}")


# ---------------------------------------------------------------------------
# Llama wrapper
# ---------------------------------------------------------------------------

def make_llama(
    gguf_path: str,
    device_idx: int,
    n_gpu_layers: int,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    seed: int,
    chat_format: Optional[str],
):
    """載入一份 Llama 實例、整顆模型 pin 到 device_idx 這張 GPU。"""
    global Llama
    if Llama is None:
        from llama_cpp import Llama as _Llama  # 延遲 import
        Llama = _Llama

    kwargs = dict(
        model_path=gguf_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        seed=seed,
        verbose=False,
        main_gpu=device_idx,
        split_mode=0,  # LLAMA_SPLIT_MODE_NONE — 整顆放在 main_gpu，不跨卡通訊
    )
    if chat_format:
        kwargs["chat_format"] = chat_format

    log(f"[INFO] 載入 GGUF 到 GPU {device_idx}: {gguf_path}")
    t0 = time.time()
    llm = Llama(**kwargs)
    log(f"[INFO] GPU {device_idx} 模型載入完成，耗時 {time.time() - t0:.1f} s "
        f"(n_ctx={n_ctx}, n_batch={n_batch}, n_gpu_layers={n_gpu_layers})")
    return llm


class LlmWorker:
    """一張 GPU + 一份 Llama 實例 + 一把獨佔鎖。

    Llama.create_completion 對單一 instance 不是 thread-safe (內部會動到
    KV cache state)；用 lock 確保同一個 instance 一次只跑一個 sample，
    但不同 instance (在不同 GPU 上) 可以同時跑。
    """

    def __init__(self, llm, device_idx: int) -> None:
        self.llm = llm
        self.device_idx = device_idx
        self.lock = threading.Lock()


def _wrap_qwen3_no_think(user_msg: str) -> str:
    """Qwen3.6 ChatML 格式 + 預先塞空 <think> 區塊，強制跳過 thinking。

    template 來源：unsloth/Qwen3.6-27B-GGUF 的 tokenizer.chat_template，當
    `enable_thinking=False` 時恰好等價於在 assistant 開頭注入
    `<think>\n\n</think>\n\n`。我們直接手刻這段，避免 llama-cpp-python
    對 chat_template_kwargs 支援不一致導致 silently 忽略的問題。
    """
    return (
        "<|im_start|>user\n"
        + user_msg
        + "\n<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def query_local_llm(
    llm,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    disable_thinking: bool = True,
) -> str:
    """跑一次推論。失敗回 '[Error] ...' 字串。

    若 disable_thinking=True (預設)：用 create_completion + 手刻 ChatML，
    並注入空 <think></think> 區塊，模型會直接吐答案。
    否則走 create_chat_completion，依 GGUF 內建 template 行為。
    """
    try:
        if disable_thinking:
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
        else:
            resp = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            choices = resp.get("choices") or []
            if not choices:
                return "[Error] no choices"
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            return content.strip() if content else "[Error] empty content"
    except Exception as exc:
        return f"[Error] inference exception: {exc}"


# ---------------------------------------------------------------------------
# Core loop
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
            "prompt": prompt,
            "predicted_answer": best_letter,
            "model_raw_response": last_response,
        }

    # 第一次 greedy；若失敗，第二次 attempt 起加溫
    for attempt in range(1, max_attempts + 1):
        t = temperature if attempt == 1 else min(temperature + 0.1 * attempt, 0.9)
        response_text = query_local_llm(
            llm, prompt, max_tokens, t, top_p,
            disable_thinking=disable_thinking,
        )
        last_response = response_text

        letter = extract_answer_letter(response_text, valid_letters)
        if letter:
            best_letter = letter
            break

        if response_text.startswith("[Error]"):
            preview = response_text[:120].replace("\n", " ")
            log(f"[WARN] id={sample_id} attempt {attempt} error: {preview}")
        elif attempt == max_attempts:
            log(f"[WARN] id={sample_id} 用盡 {max_attempts} 次仍無合法 letter；"
                f"raw={response_text[:80]!r}")

    return {
        "id": sample_id,
        "current_step": current_step,
        "prompt": prompt,
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
    chat_format: Optional[str],
    max_attempts: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
    save_every: int,
) -> None:
    samples = load_jsonl(input_path)
    log(f"載入 {len(samples)} 筆樣本：{input_path}")

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
        log("無待處理樣本，已寫出 submission/backup。")
        return

    # 為每張 GPU 載入一份模型 (1 或 2 張)
    log(f"準備在 {len(devices)} 張 GPU 上各載入一份模型: {devices}")
    workers: List[LlmWorker] = []
    for dev in devices:
        llm = make_llama(
            gguf_path=gguf_path, device_idx=dev,
            n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
            n_threads=n_threads, n_batch=n_batch,
            seed=seed, chat_format=chat_format,
        )
        workers.append(LlmWorker(llm, dev))
    n_workers = len(workers)
    log(f"[INFO] 共 {n_workers} 條 worker thread (一條對應一張 GPU)")

    results_lock = threading.Lock()
    save_lock = threading.Lock()
    completed = 0
    unresolved: List[int] = []
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
            log(f"[WARN] id={sid} (GPU {w.device_idx}) 發生例外，保留舊值: {exc}")
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex, \
            tqdm(total=len(samples), initial=skipped, desc="處理", unit="q",
                 dynamic_ncols=True) as bar:
        bar.set_postfix(miss=0)
        # round-robin 把每筆樣本分派給某張 GPU 的 worker
        future_to_sample = {
            ex.submit(_worker_fn, sample, i % n_workers): sample
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
            bar.set_postfix(miss=len(unresolved))
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
        log(f"[WARN] 仍未取到合法 letter 的 id 共 {len(unresolved)} 筆 (CSV 已 fallback)："
            f"{unresolved[:30]}" + (" ..." if len(unresolved) > 30 else ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_devices(s: str) -> List[int]:
    """parse "0" / "1" / "0,1" → [0] / [1] / [0,1]。僅允許 1 或 2 張卡。"""
    parts = [p for p in re.split(r"[,;\s]+", s.strip()) if p]
    try:
        devices = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--devices 必須是整數或以逗號分隔的整數，例如 '0' 或 '0,1' (got {s!r})"
        ) from exc
    if not devices:
        raise argparse.ArgumentTypeError("--devices 不能為空")
    if len(devices) > 2:
        raise argparse.ArgumentTypeError(
            f"本腳本只支援單卡或雙卡，--devices 最多 2 個 (got {devices})"
        )
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError(f"--devices 重複了 ({devices})")
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW4 LLM Tool Calling Agent — local GGUF runner")
    # I/O
    parser.add_argument("--input", default=os.path.join("data", "test.jsonl"),
                        help="輸入的 JSONL 檔 (預設 data/test.jsonl)")
    parser.add_argument("--output-dir", default=None,
                        help="輸出資料夾 (預設 outputs/hw4_local/{model_slug}_{mmddhhmm})")
    parser.add_argument("--output", default="submission.csv",
                        help="輸出 CSV 檔名 (預設 submission.csv)")
    parser.add_argument("--backup", default="llm_answering_results.json",
                        help="完整回應備份 JSON")
    # Model
    parser.add_argument("--gguf", required=True,
                        help="GGUF 檔絕對 / 相對路徑")
    parser.add_argument("--chat-format", default=None,
                        help="llama_cpp chat_format (預設讓 llama.cpp 自動從 GGUF metadata 偵測)")
    # GPU layout — 單卡或雙卡
    parser.add_argument("--devices", type=_parse_devices, default=[0],
                        help="要使用的 GPU index，例如 '0' (單卡) 或 '0,1' (雙卡平行)。"
                             "雙卡時每張卡會各載入一份完整模型，由 thread pool 平行分派樣本。"
                             "本腳本只支援 1 或 2 張卡。")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="每張卡 offload 到 GPU 的層數，-1 = 全部 (預設 -1)")
    # llama.cpp runtime
    parser.add_argument("--n-ctx", type=int, default=4096,
                        help="context 長度 (預設 4096，27B Q4 + KV cache 剛好塞滿單張 4090)")
    parser.add_argument("--n-threads", type=int, default=8,
                        help="CPU thread 數 (預設 8)")
    parser.add_argument("--n-batch", type=int, default=512,
                        help="prompt batch tokens (預設 512)")
    parser.add_argument("--seed", type=int, default=42)
    # Decoding
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="模型回應最多 token (預設 512，雖然停用 thinking 但模型仍會寫推理散文，需給足空間到結尾 Answer 行)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--max-attempts", type=int, default=4,
                        help="同一筆最多重試幾次 (預設 4)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="預設關閉 Qwen3.6 thinking 模式；指定此 flag 才打開 thinking")
    parser.add_argument("--save-every", type=int, default=20,
                        help="每完成 N 筆 flush 到磁碟 (預設 20)")
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()

    output_dir = args.output_dir or default_output_dir(args.gguf)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = args.output if os.path.isabs(args.output) else os.path.join(output_dir, args.output)
    backup_json = args.backup if os.path.isabs(args.backup) else os.path.join(output_dir, args.backup)

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
        chat_format=args.chat_format,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        disable_thinking=not args.enable_thinking,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
