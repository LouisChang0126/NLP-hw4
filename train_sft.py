"""
HW4 SFT 訓練腳本：QLoRA on Qwen3.6-27B (對應 spec Q2 #3 / #4)。

設計
====
- 底座：`unsloth/Qwen3.6-27B` (FP16 safetensors) → 用 bitsandbytes 4-bit load
  (NF4 + double quant)，單張 24GB GPU 就能塞下 27B + LoRA adapter + activation。
- LoRA：r=16, alpha=32, dropout=0.05，target attn + MLP 線性層。
- 資料：train.jsonl 全部 (~13.5k 筆) — 對應 spec 規定 SFT 用 train data；
  dev split (由 run_dev_eval.py 切的 data/dev.jsonl) 排除掉，避免污染。
- Prompt mode：與 run_dev_eval.py 用同一份 build_prompt，
  SFT × Full / SFT × Struct 只差 --prompt-mode。
- Loss：只對 "Answer: X" 那段算 loss (input 區塊 mask 掉)；用 trl.SFTTrainer 的
  response_template 機制達成。

執行：
  # SFT × Full-Info
  CUDA_VISIBLE_DEVICES=0 python train_sft.py --prompt-mode full \
      --output-dir outputs/sft_full

  # SFT × Structural-Only
  CUDA_VISIBLE_DEVICES=1 python train_sft.py --prompt-mode struct \
      --output-dir outputs/sft_struct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Set

# 提前用 env 鎖卡，避免 torch 把所有 GPU 都看見
# (若 CLI 沒設 CUDA_VISIBLE_DEVICES，後面 --device 參數會補)
# Note: 這支腳本「一次只用一張卡」，並行訓練時各自跑一份 process
# 一個 process / 一張卡。

# 重用 run_dev_eval 的 prompt builder，保持訓練 / 評估完全對齊
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_dev_eval import (  # noqa: E402
    build_prompt,
    load_jsonl,
    log,
)


def _build_dataset_records(
    train_path: str,
    dev_path: str,
    prompt_mode: str,
) -> List[dict]:
    """讀 train.jsonl，排除 dev split 的 id，產生 (prompt, target) pairs。"""
    train = load_jsonl(train_path)
    if os.path.exists(dev_path):
        dev = load_jsonl(dev_path)
        dev_ids = {s["id"] for s in dev}
        log(f"[INFO] 從訓練集排除 dev split {len(dev_ids)} 筆")
    else:
        dev_ids = set()
        log(f"[WARN] 找不到 dev split {dev_path}，整份 train.jsonl 都用來訓練")

    train_used = [s for s in train if s["id"] not in dev_ids]
    log(f"[INFO] 訓練集樣本數: {len(train_used)} (原 train.jsonl {len(train)} 筆)")

    records: List[dict] = []
    skipped = 0
    for s in train_used:
        gold = s.get("answer", "")
        options = s.get("options", {}) or {}
        if gold not in options:
            skipped += 1
            continue
        prompt = build_prompt(
            s.get("full_context", ""),
            s.get("current_step", ""),
            options,
            prompt_mode,
        )
        # 訓練 target：與 run_dev_eval 抽答案的正則對齊
        target = f"Answer: {gold}"
        records.append({"prompt": prompt, "target": target})
    if skipped:
        log(f"[WARN] 跳過 {skipped} 筆 (gold answer 不在 options keys 中)")
    return records


# Qwen3 ChatML 包裝。訓練樣本切成 (prompt_text, completion_text)：
#   prompt_text     = user msg + assistant 起始 + 空 <think></think>
#                     ← 這段在 loss 中會被 mask (只當 context)
#   completion_text = "Answer: <X><|im_end|>"
#                     ← 只對這幾個 token 算 loss
# trl 1.0 的 SFTTrainer 看到 "prompt" + "completion" 兩個 column 會自動套
# completion-only loss masking。
def _format_for_training(prompt: str, target: str) -> dict:
    return {
        "prompt": (
            "<|im_start|>user\n"
            + prompt
            + "\n<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        "completion": target + "<|im_end|>",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HW4 SFT training (QLoRA on Qwen3.6-27B)"
    )
    parser.add_argument("--base-model", default="unsloth/Qwen3.6-27B",
                        help="HuggingFace repo id 或本地路徑")
    parser.add_argument("--train-input", default=os.path.join("data", "train.jsonl"))
    parser.add_argument("--dev-input", default=os.path.join("data", "dev.jsonl"),
                        help="dev split 路徑 (用來排除)；不存在則使用整份 train")
    parser.add_argument("--prompt-mode", required=True, choices=["full", "struct"])
    parser.add_argument("--output-dir", required=True,
                        help="模型 / log / adapter 寫出位置")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="effective batch = batch_size × grad_accum")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, default=3000,
                        help="從 train.jsonl 取最多這麼多筆訓練 (預設 3000，加速用)")
    parser.add_argument("--lora-attn-only", action="store_true", default=True,
                        help="LoRA 只掛 attention (q/k/v/o)，省 memory + 加速")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.makedirs(args.output_dir, exist_ok=True)
    log(f"輸出資料夾: {args.output_dir}")
    log(f"prompt_mode = {args.prompt_mode}")

    # ── 載入 base ─────────────────────────────────
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    log(f"[INFO] 載入 tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    log(f"[INFO] 載入 base model in 4-bit (NF4 + double quant)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},  # 整顆裝在當前可見 GPU 的 device 0
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    # 不走 peft.prepare_model_for_kbit_training — 在 peft 0.18 上會把 4-bit base
    # 部分 weights 升回 FP32，27B 直接 OOM。改手動做必要設定：
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()  # 讓 grad checkpointing 能正常 backward
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if not args.lora_attn_only:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── 準備資料集 ────────────────────────────────
    records = _build_dataset_records(args.train_input, args.dev_input, args.prompt_mode)
    log(f"[INFO] 候選訓練樣本 {len(records)} 筆")
    if args.max_train_samples and len(records) > args.max_train_samples:
        import random
        random.Random(args.seed).shuffle(records)
        records = records[:args.max_train_samples]
        log(f"[INFO] 取 {len(records)} 筆作訓練 (--max-train-samples)")

    pairs = [_format_for_training(r["prompt"], r["target"]) for r in records]
    ds = Dataset.from_list(pairs)

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        max_length=args.max_seq_len,
        packing=False,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        completion_only_loss=True,  # 自動 mask prompt 段，只對 completion 算 loss
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=sft_cfg,
    )

    log("[INFO] 開始訓練...")
    trainer.train()
    log("[INFO] 訓練結束，存 LoRA adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # 寫一份 meta 方便 inference 時知道用哪個 prompt mode
    with open(os.path.join(args.output_dir, "sft_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.base_model,
            "prompt_mode": args.prompt_mode,
            "epochs": args.epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "n_train_examples": len(records),
        }, f, ensure_ascii=False, indent=2)
    log(f"[INFO] 完成。adapter + tokenizer → {args.output_dir}")


if __name__ == "__main__":
    main()
