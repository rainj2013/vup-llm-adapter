#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用推理脚本（Qwen 家族可用）：
1) 纯基座（不加 LoRA）
2) 基座 + LoRA（PEFT 方式）
3) 合并后的全量模型（像普通 HF 模型一样加载）

更新点：
- 自动决定 do_sample：当 temperature>0 或 top_p<1.0 或 top_k>0 时启用采样
- 仅在采样启用时传入 temperature/top_p/top_k，避免无效参数告警
- 继续支持本地优先加载、dtype/torch_dtype 兼容、损坏文件诊断、离线/强制在线开关
"""

import argparse
import inspect
import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- dtype 选择 & from_pretrained 兼容 ----------------

def pick_dtype(use_fp16: bool) -> torch.dtype:
    if torch.cuda.is_available() and use_fp16:
        return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    return torch.float32

def from_pretrained_compat(cls, model_path, dtype, **kw):
    sig = inspect.signature(cls.from_pretrained)
    params = dict(kw)
    if "dtype" in sig.parameters:
        params["dtype"] = dtype
    else:
        params["torch_dtype"] = dtype
    return cls.from_pretrained(model_path, **params)

# ---------------- 诊断工具：打印本地关键文件大小 ----------------

def diag_local_files(model_id_or_path: str, names: list):
    print("[DIAG] 检测本地关键文件：")
    if os.path.isdir(model_id_or_path):
        for name in names:
            p = os.path.join(model_id_or_path, name)
            if os.path.exists(p):
                try:
                    sz = os.path.getsize(p)
                    print(f"       - {p} ({sz} bytes)")
                except Exception:
                    print(f"       - {p} (无法获取大小)")
    try:
        from huggingface_hub import hf_hub_download
        for name in names:
            try:
                p = hf_hub_download(model_id_or_path, name, local_files_only=True)
                if os.path.exists(p):
                    sz = os.path.getsize(p)
                    print(f"       - {p} ({sz} bytes)")
            except Exception:
                pass
    except Exception:
        pass
    print("       如果看到 size=0 或明显异常，请删除该文件并重新下载。")

# ---------------- 分词器加载（本地优先/联网回退/诊断） ----------------

def load_tokenizer(model_path: str, use_fast=True, trust_remote_code=True,
                   offline_only=False, force_online=False):
    def _load(local_only: bool):
        return AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast, trust_remote_code=trust_remote_code,
            local_files_only=local_only
        )

    if offline_only:
        try:
            tok = _load(local_only=True)
            print("[INFO] Tokenizer loaded (offline only).")
            return tok
        except json.JSONDecodeError:
            print("\n[ERROR] tokenizer_config.json 解析失败（JSONDecodeError, offline only）。")
            diag_local_files(model_path, ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"])
            raise
        except Exception as e:
            print(f"[ERROR] 本地加载分词器失败：{e}")
            diag_local_files(model_path, ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"])
            raise

    if force_online:
        try:
            tok = _load(local_only=False)
            print("[INFO] Tokenizer loaded (force online).")
            return tok
        except json.JSONDecodeError:
            print("\n[ERROR] tokenizer_config.json 解析失败（JSONDecodeError, online）。")
            print("可能是镜像/代理返回 HTML/空响应。建议切换网络或使用 HF 镜像/直连。")
            raise
        except Exception as e:
            print(f"[ERROR] 联网加载分词器失败：{e}")
            raise

    try:
        tok = _load(local_only=True)
        print("[INFO] Tokenizer loaded (local cache only).")
        return tok
    except Exception as e_local:
        print(f"[WARN] Tokenizer local-only failed: {e_local}\n[INFO] Fallback to network...")
        try:
            tok = _load(local_only=False)
            print("[INFO] Tokenizer loaded (network fallback).")
            return tok
        except json.JSONDecodeError:
            print("\n[ERROR] tokenizer_config.json 解析失败（JSONDecodeError）。")
            diag_local_files(model_path, ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"])
            print("\n修复建议：\n"
                  "  1) 若本地目录损坏，删除坏文件后只重下该文件；\n"
                  "  2) 若走缓存：huggingface-cli delete-cache --pattern '<模型名或关键字>'；\n"
                  "  3) 国内加速：export HF_ENDPOINT=https://hf-mirror.com；\n"
                  "  4) 完整离线再加载：huggingface-cli download <repo> --local-dir ./models/<name> "
                  "--local-dir-use-symlinks False --resume-download --force-download")
            raise
        except Exception as e_net:
            print(f"[ERROR] Tokenizer network fallback failed: {e_net}")
            raise

# ---------------- 模型加载（本地优先/联网回退/诊断） ----------------

def load_model(model_path: str, dtype: torch.dtype, peft_dir: str = None,
               trust_remote_code=True, offline_only=False, force_online=False):
    def _load(local_only: bool):
        if peft_dir:
            from peft import PeftModel
            base = from_pretrained_compat(
                AutoModelForCausalLM, model_path, dtype,
                device_map="auto", trust_remote_code=trust_remote_code,
                local_files_only=local_only
            )
            return PeftModel.from_pretrained(base, peft_dir, device_map="auto", local_files_only=local_only)
        else:
            return from_pretrained_compat(
                AutoModelForCausalLM, model_path, dtype,
                device_map="auto", trust_remote_code=trust_remote_code,
                local_files_only=local_only
            )

    if offline_only:
        try:
            m = _load(local_only=True); m.eval()
            print(f"[INFO] Loaded (offline only): {model_path}" + (f" + {peft_dir}" if peft_dir else ""))
            return m
        except json.JSONDecodeError:
            print("\n[ERROR] 模型索引 JSONDecodeError（offline only）")
            diag_local_files(model_path, ["model.safetensors.index.json", "pytorch_model.bin.index.json"])
            raise
        except Exception as e:
            print(f"[ERROR] 本地加载模型失败：{e}")
            raise

    if force_online:
        try:
            m = _load(local_only=False); m.eval()
            print(f"[INFO] Loaded (force online): {model_path}" + (f" + {peft_dir}" if peft_dir else ""))
            return m
        except json.JSONDecodeError:
            print("\n[ERROR] 模型索引 JSONDecodeError（online）。可能是镜像/代理返回 HTML。")
            raise
        except Exception as e:
            print(f"[ERROR] 联网加载模型失败：{e}")
            raise

    try:
        m = _load(local_only=True); m.eval()
        print(f"[INFO] Loaded (local cache only): {model_path}" + (f" + {peft_dir}" if peft_dir else ""))
        return m
    except Exception as e_local:
        print(f"[WARN] Local-only load failed: {e_local}\n[INFO] Fallback to network...")
        try:
            m = _load(local_only=False); m.eval()
            print(f"[INFO] Loaded (network fallback): {model_path}" + (f" + {peft_dir}" if peft_dir else ""))
            return m
        except json.JSONDecodeError:
            print("\n[ERROR] 模型索引 JSONDecodeError。")
            diag_local_files(model_path, ["model.safetensors.index.json", "pytorch_model.bin.index.json"])
            print("\n修复建议：\n"
                  "  1) 删除 0 字节/异常的 *.index.json 后只重下该文件；\n"
                  "  2) delete-cache 定向清理；\n"
                  "  3) 完整离线到 ./models/<name> 再加载；\n"
                  "  4) 国内可设置 HF_ENDPOINT=https://hf-mirror.com。")
            raise
        except Exception as e_net:
            print(f"[ERROR] 模型网络回退失败：{e_net}")
            raise

# ---------------- 输入构造 ----------------

def build_inputs(tokenizer, messages, use_chat_template: bool, device):
    if use_chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt").to(device)
    else:
        joined = ""
        for m in messages:
            joined += f"<|{m['role']}|>\n{m['content']}\n"
        joined += tokenizer.eos_token or ""
        return tokenizer(joined, return_tensors="pt").to(device)

# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 仓库名或本地路径（基座或合并后全量皆可）")
    ap.add_argument("--peft_dir", default=None, help="可选：LoRA/PEFT 目录（叠加在基座上推理）")
    ap.add_argument("--prompt", required=True, help="用户输入内容")
    ap.add_argument("--system", default="你是一个客观、简洁的中文助理。", help="可选：system 提示词")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0, help="事实问答建议 0 更稳定")
    ap.add_argument("--top_p", type=float, default=1.0, help="与 temperature 搭配，采样时生效")
    ap.add_argument("--top_k", type=int, default=0, help=">0 时启用 top-k 采样")
    ap.add_argument("--use_chat_template", action="store_true", help="使用 chat 模板（推荐）")
    ap.add_argument("--no_fp16", action="store_true", help="禁用 fp16/bf16，强制 float32")
    ap.add_argument("--offline_only", action="store_true", help="仅使用本地缓存/本地目录（不联网）")
    ap.add_argument("--force_online", action="store_true", help="跳过本地，强制联网加载（不建议常用）")
    args = ap.parse_args()

    if args.offline_only and args.force_online:
        print("[ERROR] --offline_only 与 --force_online 不能同时使用。")
        sys.exit(1)

    # 分词器
    tokenizer = load_tokenizer(
        args.model, use_fast=True, trust_remote_code=True,
        offline_only=args.offline_only, force_online=args.force_online
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 模型
    dtype = pick_dtype(use_fp16=not args.no_fp16)
    model = load_model(
        args.model, dtype=dtype, peft_dir=args.peft_dir, trust_remote_code=True,
        offline_only=args.offline_only, force_online=args.force_online
    )

    # 构造消息
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    inputs = build_inputs(tokenizer, messages, args.use_chat_template, model.device)

    # 是否启用采样
    use_sampling = (args.temperature > 0.0) or (args.top_p < 1.0) or (args.top_k > 0)

    # 生成参数（仅在采样时传入 temperature/top_p/top_k）
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": use_sampling,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if use_sampling:
        # 只在采样启用时才添加这些字段，避免无效参数告警
        gen_kwargs["temperature"] = args.temperature if args.temperature > 0.0 else 1.0
        gen_kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n===== MODEL OUTPUT =====\n")
    print(text)
    print("\n========================\n")

if __name__ == "__main__":
    main()

