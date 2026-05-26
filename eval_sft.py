"""
HW4 SFT 評估腳本：載入 base + LoRA adapter，在 dev split 上算 accuracy +
做 Q3 ambiguity 錯誤分析。

對齊 run_dev_eval.py 的 prompt 與評估邏輯，差別只在後端是
transformers + peft (base 4-bit + LoRA) 而非 llama.cpp。

用法 (conda env: NLP2)：
  CUDA_VISIBLE_DEVICES=0 python eval_sft.py \
      --adapter outputs/sft_full \
      --prompt-mode full \
      --output-dir outputs/exp_dev_sft_full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Set

from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_dev_eval import (  # noqa: E402
    build_prompt,
    extract_answer_letter,
    build_or_load_dev_split,
    evaluate_and_analyze,
    _build_submission_rows,
    save_outputs,
    load_existing_results,
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


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="HW4 SFT dev eval (base + LoRA adapter via transformers)"
    )
    parser.add_argument("--adapter", required=True,
                        help="LoRA adapter 資料夾 (train_sft.py 的 --output-dir)")
    parser.add_argument("--base-model", default=None,
                        help="base 模型 (預設讀 adapter/adapter_config.json 的 base_model_name_or_path)")
    parser.add_argument("--prompt-mode", required=True, choices=["full", "struct"])
    parser.add_argument("--dev-input", default=os.path.join("data", "dev.jsonl"))
    parser.add_argument("--dev-size", type=int, default=1000,
                        help="若 dev_input 不存在則從 train.jsonl 切此筆數")
    parser.add_argument("--train-input", default=os.path.join("data", "train.jsonl"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="SFT 後模型只需要吐 'Answer: X' 幾個 token (預設 8)")
    parser.add_argument("--ambiguity-threshold", type=float, default=0.4)
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    submission_csv = os.path.join(args.output_dir, "dev_submission.csv")
    backup_json = os.path.join(args.output_dir, "llm_answering_results.json")
    eval_json = os.path.join(args.output_dir, "eval_results.json")
    log(f"輸出資料夾: {args.output_dir}")

    dev_samples = build_or_load_dev_split(args.train_input, args.dev_input, args.dev_size)

    # ── 載入模型 ──
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    base_model = args.base_model
    if base_model is None:
        cfg_path = os.path.join(args.adapter, "adapter_config.json")
        if not os.path.exists(cfg_path):
            log(f"[FATAL] {cfg_path} 不存在")
            sys.exit(1)
        with open(cfg_path) as f:
            cfg = json.load(f)
        base_model = cfg.get("base_model_name_or_path")
        log(f"[INFO] 自動偵測 base_model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    log(f"[INFO] 載入 base 4-bit: {base_model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    log(f"[INFO] base 載入完成，耗時 {time.time()-t0:.1f} s")
    log(f"[INFO] 掛 LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    log(f"[INFO] 模型總體載入耗時 {time.time()-t0:.1f} s")

    # ── 推論 ──
    existing = load_existing_results(backup_json)
    results_by_id: Dict[int, dict] = dict(existing)
    to_process = []
    skipped = 0
    for s in dev_samples:
        sid = s["id"]
        prev = results_by_id.get(sid)
        prev_letter = (prev or {}).get("predicted_answer", "") or ""
        prev_mode = (prev or {}).get("prompt_mode", "")
        valid_letters = set((s.get("options") or {}).keys())
        if prev_letter in valid_letters and prev_mode == args.prompt_mode:
            skipped += 1
        else:
            to_process.append(s)
    log(f"沿用 {skipped}，需推論 {len(to_process)}")

    completed = 0
    for sample in tqdm(to_process, desc=f"SFT-{args.prompt_mode}", unit="q", dynamic_ncols=True):
        sid = sample["id"]
        options = sample.get("options", {}) or {}
        valid_letters = set(options.keys())
        prompt = build_prompt(
            sample.get("full_context", ""),
            sample.get("current_step", ""),
            options,
            args.prompt_mode,
        )
        wrapped = _wrap_qwen3_no_think(prompt)
        inputs = tokenizer(wrapped, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        letter = extract_answer_letter(text, valid_letters)

        results_by_id[sid] = {
            "id": sid,
            "current_step": sample.get("current_step", ""),
            "prompt_mode": args.prompt_mode,
            "predicted_answer": letter,
            "model_raw_response": text,
        }

        completed += 1
        if completed % args.save_every == 0 or completed == len(to_process):
            rows = _build_submission_rows(dev_samples, results_by_id)
            try:
                save_outputs(submission_csv, backup_json, rows, results_by_id)
            except Exception as exc:
                log(f"[WARN] flush 失敗: {exc}")

    rows = _build_submission_rows(dev_samples, results_by_id)
    save_outputs(submission_csv, backup_json, rows, results_by_id)

    # ── 評估 ──
    log("\n" + "=" * 60)
    stats = evaluate_and_analyze(results_by_id, dev_samples, args.ambiguity_threshold)
    log(f"prompt_mode = {args.prompt_mode} (SFT)")
    log(f"  accuracy       = {stats['accuracy']:.4f}  ({stats['correct']}/{stats['total']})")
    log(f"  ambiguity err  = {stats['ambiguity_errors']}/{stats['errors']} "
        f"({stats['ambiguity_pct_of_errors']*100:.1f}% of errors)")

    stats["prompt_mode"] = args.prompt_mode
    stats["adapter"] = args.adapter
    stats["base_model"] = base_model
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log(f"寫出評估結果: {eval_json}")


if __name__ == "__main__":
    main()
