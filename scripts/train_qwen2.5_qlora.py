#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA fine-tune for Qwen (默认：Qwen/Qwen2.5-7B-Instruct)

示例：
python scripts/train_qwen3_qlora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --data_file outputs/vtuber_qa.jsonl \
  --out_dir qwen2p5-vtuber-qlora \
  --max_seq_len 2048 --epochs 2 --lr 2e-4 \
  --batch_size 1 --grad_accum 16 \
  --lora_r 32 --lora_alpha 16 --lora_dropout 0.05

国内网络建议：
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
  然后 --base_model ./models/Qwen2.5-7B-Instruct
"""

import os, re, json, argparse

# ---- Transformers 版本检查（>=4.46，使用 eval_strategy）----
MIN_TF_VER_DEFAULT = "4.46.0"

def _parse_ver(s: str):
    nums = re.findall(r"\d+", s)
    nums = (nums + ["0","0","0"])[:3]
    return tuple(int(x) for x in nums)

def ensure_transformers(min_ver: str):
    try:
        import transformers
    except Exception as e:
        raise RuntimeError(
            "未安装 transformers，请先：uv add transformers accelerate peft bitsandbytes datasets"
        ) from e
    have = transformers.__version__
    if _parse_ver(have) < _parse_ver(min_ver):
        raise RuntimeError(
            f"transformers 版本过低：{have}，需 ≥ {min_ver}。"
            f"请升级：uv add 'transformers>={min_ver}'"
        )

def print_hf_tips():
    print(
        "\n[提示] 如遇下载缓慢/403：\n"
        "  1) 临时镜像：export HF_ENDPOINT=https://hf-mirror.com\n"
        "  2) 提前下载：huggingface-cli download <repo> --local-dir ./models/<name>\n"
        "  3) 然后将 --base_model 指到本地路径\n"
    )

def main():
    ap = argparse.ArgumentParser()
    # 基本
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct", help="HF repo 或本地路径")
    ap.add_argument("--data_file", required=True, help="JSONL，每行形如 {'messages': [...]}")
    ap.add_argument("--out_dir", default="qwen2p5-qlora-out")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    # LoRA
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="*", default=None,
                    help="LoRA 作用层(留空自动匹配常见 linear 层)")
    # 训练细节
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--no_fp16", action="store_true", help="禁用 fp16/bf16 自动选择")
    # 版本检查
    ap.add_argument("--min_tf_ver", default=MIN_TF_VER_DEFAULT, help="最小 transformers 版本要求")
    args = ap.parse_args()

    ensure_transformers(args.min_tf_ver)

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling, BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # bf16/fp16 自动选择
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 and not args.no_fp16
    fp16 = (not bf16) and torch.cuda.is_available() and not args.no_fp16

    # QLoRA: 4bit 量化加载
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32,
    )

    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            use_fast=True,
            trust_remote_code=True
        )
    except Exception as e:
        print("[ERROR] 加载 tokenizer 失败：", repr(e))
        print_hf_tips()
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（4bit 量化）
    try:
        print(f"[INFO] 加载模型：{args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print("[ERROR] 加载模型失败：", repr(e))
        print_hf_tips()
        raise

    # 准备 QLoRA
    model = prepare_model_for_kbit_training(model)
    lora_modules = args.target_modules or [
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_modules
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 数据
    raw_ds = load_dataset("json", data_files=args.data_file, split="train")

    def apply_chat_template(messages):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = []
            for m in messages:
                role = m.get("role","")
                content = m.get("content","")
                if role == "system":
                    text.append(f"<|system|>\n{content}\n")
                elif role == "user":
                    text.append(f"<|user|>\n{content}\n")
                elif role == "assistant":
                    text.append(f"<|assistant|>\n{content}\n")
            return "".join(text) + (tokenizer.eos_token or "")

    # —— 兼容一维/二维的 labels 处理 —— #
    def preprocess(ex):
        text = apply_chat_template(ex["messages"])
        tokenized = tokenizer(
            text,
            max_length=args.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        pad_id = tokenizer.pad_token_id
        ids = tokenized["input_ids"]

        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            labels = [[-100 if tok == pad_id else tok for tok in seq] for seq in ids]
        else:
            labels = [-100 if tok == pad_id else tok for tok in ids]

        tokenized["labels"] = labels
        return tokenized

    proc_ds = raw_ds.map(preprocess, remove_columns=raw_ds.column_names, desc="Tokenizing")

    os.makedirs(args.out_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=fp16,
        bf16=bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        # 这里用 eval_strategy（evaluation_strategy 在 4.46+ 已移除）
        eval_strategy="no" if args.eval_steps <= 0 else "steps",
        eval_steps=None if args.eval_steps <= 0 else args.eval_steps,
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=proc_ds,
        eval_dataset=None,
        data_collator=data_collator,
    )

    print("[INFO] 开始训练 ...")
    trainer.train()
    print("[INFO] 训练完成，保存 LoRA 权重到：", args.out_dir)
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print(
        "\n[可选后续] 合并 LoRA 为全量权重（便于部署/转 GGUF）：\n"
        "  python scripts/merge_lora.py --base_model <HF或本地路径> --lora_dir {} --out_dir merged-qwen\n"
        "或直接以 PEFT 方式加载推理。".format(args.out_dir)
    )

if __name__ == "__main__":
    main()

