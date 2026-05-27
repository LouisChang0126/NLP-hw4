"""
HW4 SFT 推論腳本 (test set → submission.csv)。

對齊 local_llm_inference.py 的雙卡平行設計，但後端換成 transformers + peft adapter：
每張 GPU 各載入一份 (base 4-bit + LoRA adapter) instance，ThreadPoolExecutor
round-robin 分派樣本。Prompt + parser 與 run_dev_eval.py 共用。

用法：
  python predict_sft.py \
      --adapter outputs/sft_full \
      --prompt-mode full \
      --devices 0,1 \
      --input data/test.jsonl \
      --output-dir outputs/test_sft_full
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from typing import Dict, List, Optional, Set

from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_dev_eval import (  # noqa: E402
    build_prompt,
    extract_answer_letter,
    load_jsonl,
    load_existing_results,
    save_outputs,
    _build_submission_rows,
    log,
    _configure_stdout,
)


def _wrap_qwen3_no_think(user_msg: str) -> str:
    return (
        "<|im_start|>user\n"
        + user_msg
        + "\n<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _parse_devices(s: str) -> List[int]:
    parts = [p for p in re.split(r"[,;\s]+", s.strip()) if p]
    devices = [int(p) for p in parts]
    if not devices or len(devices) > 2 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError(f"--devices 必須是 1 或 2 個唯一整數 (got {s!r})")
    return devices


class SftWorker:
    """一張 GPU + 一份 (base + adapter) + 一把獨佔鎖。"""

    def __init__(self, model, tokenizer, device_idx: int, max_new_tokens: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device_idx = device_idx
        self.lock = threading.Lock()
        self.max_new_tokens = max_new_tokens


def _load_worker(
    base_model: str,
    adapter: str,
    device_idx: int,
    max_new_tokens: int,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    log(f"[INFO] GPU {device_idx}: 載入 base 4-bit + adapter")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map={"": device_idx},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter, device_map={"": device_idx})
    model.eval()
    log(f"[INFO] GPU {device_idx}: 載入完成，耗時 {time.time()-t0:.1f}s")
    return SftWorker(model, tok, device_idx, max_new_tokens)


def _generate(worker: SftWorker, prompt: str, valid_letters: Set[str]) -> tuple[str, str]:
    import torch
    wrapped = _wrap_qwen3_no_think(prompt)
    inputs = worker.tokenizer(wrapped, return_tensors="pt").to(worker.model.device)
    with torch.no_grad():
        out = worker.model.generate(
            **inputs,
            max_new_tokens=worker.max_new_tokens,
            do_sample=False,
            pad_token_id=worker.tokenizer.pad_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    text = worker.tokenizer.decode(gen, skip_special_tokens=True)
    letter = extract_answer_letter(text, valid_letters)
    return letter, text


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="HW4 SFT predict on test set (dual-GPU)")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default=None,
                        help="預設讀 adapter/adapter_config.json 的 base_model_name_or_path")
    parser.add_argument("--prompt-mode", required=True, choices=["full", "struct"])
    parser.add_argument("--devices", type=_parse_devices, default=[0, 1])
    parser.add_argument("--input", default=os.path.join("data", "test.jsonl"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    base_model = args.base_model
    if base_model is None:
        cfg_path = os.path.join(args.adapter, "adapter_config.json")
        with open(cfg_path) as f:
            base_model = json.load(f).get("base_model_name_or_path")
        log(f"[INFO] 自動偵測 base_model: {base_model}")

    os.makedirs(args.output_dir, exist_ok=True)
    submission_csv = os.path.join(args.output_dir, "submission.csv")
    backup_json = os.path.join(args.output_dir, "llm_answering_results.json")
    log(f"輸出資料夾: {args.output_dir}")

    samples = load_jsonl(args.input)
    log(f"載入 {len(samples)} 筆樣本：{args.input}")

    existing = load_existing_results(backup_json)
    if existing:
        log(f"偵測到既有結果 {len(existing)} 筆")
    results_by_id: Dict[int, dict] = dict(existing)

    to_process = []
    skipped = 0
    for s in samples:
        sid = s["id"]
        prev = results_by_id.get(sid)
        prev_letter = (prev or {}).get("predicted_answer", "") or ""
        prev_mode = (prev or {}).get("prompt_mode", "")
        valid_letters = set((s.get("options") or {}).keys())
        if prev_letter in valid_letters and prev_mode == args.prompt_mode:
            skipped += 1
        else:
            to_process.append(s)
    log(f"沿用 {skipped}，待推論 {len(to_process)}")

    if not to_process:
        rows = _build_submission_rows(samples, results_by_id)
        save_outputs(submission_csv, backup_json, rows, results_by_id)
        log("無待處理樣本。"); return

    workers: List[SftWorker] = []
    for dev in args.devices:
        workers.append(_load_worker(base_model, args.adapter, dev, args.max_new_tokens))
    n_workers = len(workers)
    log(f"[INFO] 共 {n_workers} 條 worker thread")

    results_lock = threading.Lock()
    save_lock = threading.Lock()
    completed = 0
    total = len(to_process)

    def _job(sample: dict, worker_idx: int) -> str:
        sid = sample["id"]
        w = workers[worker_idx]
        options = sample.get("options", {}) or {}
        valid_letters = set(options.keys())
        prompt = build_prompt(
            sample.get("full_context", ""),
            sample.get("current_step", ""),
            options,
            args.prompt_mode,
        )
        try:
            with w.lock:
                letter, raw = _generate(w, prompt, valid_letters)
        except Exception as exc:  # pragma: no cover
            log(f"[WARN] id={sid} (GPU {w.device_idx}) 例外: {exc}")
            letter, raw = "", f"[Error] {exc}"
        result = {
            "id": sid,
            "current_step": sample.get("current_step", ""),
            "prompt_mode": args.prompt_mode,
            "predicted_answer": letter,
            "model_raw_response": raw,
        }
        with results_lock:
            results_by_id[sid] = result
        return letter

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex, \
            tqdm(total=len(samples), initial=skipped, desc=f"SFT-{args.prompt_mode}",
                 unit="q", dynamic_ncols=True) as bar:
        futures = {
            ex.submit(_job, sample, i % n_workers): sample
            for i, sample in enumerate(to_process)
        }
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                fut.result()
            except Exception as exc:  # pragma: no cover
                log(f"[WARN] worker fail: {exc}")
            bar.update(1)
            if completed % args.save_every == 0 or completed == total:
                with save_lock, results_lock:
                    rows = _build_submission_rows(samples, results_by_id)
                    try:
                        save_outputs(submission_csv, backup_json, rows, results_by_id)
                    except Exception as exc:
                        log(f"[WARN] flush 失敗: {exc}")

    with save_lock, results_lock:
        rows = _build_submission_rows(samples, results_by_id)
        save_outputs(submission_csv, backup_json, rows, results_by_id)
    log(f"\n完成。輸出: {submission_csv}")


if __name__ == "__main__":
    main()
