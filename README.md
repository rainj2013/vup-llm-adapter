# vtuber-qlora (个人学习用, 使用 uv 管理环境)

一套在 **单卡 RTX 5090** 上跑通的端到端工程：
- 通过 MediaWiki API 抓取萌娘百科的 Vtuber 相关条目
- 解析/清洗 wikitext，结构化为 `{title, info, text}`
- 生成 **Qwen** 聊天式 **SFT 问答样本**（JSONL）
- 使用 **QLoRA(4bit) + TRL** 对 **Qwen2.5-7B-Instruct** 进行指令微调
- 通过 **PEFT 适配器** 推理验证

> ⚠️ 本工程用于个人学习演示。你在实际使用中请**自行处理许可与署名合规**。

---

## 目录结构
```
vtuber-qlora/
├── pyproject.toml
├── .gitignore
├── data/
│   ├── raw_pages/          # 抓取的 .wikitext 原文
│   └── clean_json/         # 清洗后的结构化 JSON
├── outputs/
│   ├── vtuber_qa.jsonl     # 规则脚本生成的问答数据
│   └── vtuber_qa.from_llm.jsonl # LLM 自动撰写的问答数据
└── scripts/
    ├── fetch_pages.py
    ├── clean_and_struct.py
    ├── build_qa_jsonl.py
    ├── extract_with_llm.py
    ├── train_qwen2.5_qlora.py
    └── infer.py
```

## 0. 环境准备（使用 uv）
推荐 Python 3.10+，Ubuntu22.04 (WSL2) 内执行。

安装 uv：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

创建虚拟环境：
```bash
uv venv .venv
source .venv/bin/activate
```

安装依赖（会根据 `pyproject.toml` 自动同步）：
```bash
uv sync
```

## 1. 抓取数据（MediaWiki API）
```bash
python scripts/fetch_pages.py   --base https://zh.moegirl.org.cn/api.php   --category "Category:在bilibili活动过的虚拟UP主"   --outdir data/raw_pages   --sleep 0.2 --resume
```

## 2. 清洗与结构化
```bash
python scripts/clean_and_struct.py   --indir data/raw_pages   --outdir data/clean_json
```

## 3. 生成问答样本（JSONL）
纯文本处理生成（基于规则抽取，字段稳定、便于批量生成，输出 `outputs/vtuber_qa.jsonl`）：
```bash
python scripts/build_qa_jsonl.py   --indir data/clean_json   --outfile outputs/vtuber_qa.jsonl   --base_url https://zh.moegirl.org.cn
```

利用 llm 生成（以 MOONSHOT 的 API 为例，问答由模型自动撰写，输出 `outputs/vtuber_qa.from_llm.jsonl`）：
```bash
export MOONSHOT_API_KEY=sk-xxxxxxxx
# 可选：自定义基址
# export MOONSHOT_BASE_URL=https://api.moonshot.cn/v1

python scripts/extract_with_llm.py \
  --provider moonshot \
  --model kimi-k2-0905-preview \
  --indir data/clean_json \
  --outdir data/extracted_json \
  --gen_qa \
  --qa_out outputs/vtuber_qa.from_llm.jsonl \
  --rpm 20 --max_retries 8 --backoff_base 2.0
```

> 备注：`outputs/vtuber_qa.jsonl` 来自规则脚本（可复现性强）；`outputs/vtuber_qa.from_llm.jsonl` 由 LLM 生成（答案更灵活，可包含补充信息）。

## 4. 用 QLoRA 训练（PEFT）
```bash
python scripts/train_qwen2.5_qlora.py   --base_model Qwen/Qwen2.5-7B-Instruct   --data_file outputs/vtuber_qa.jsonl   --out_dir qwen25-vtuber-qlora   --max_seq_len 2048   --epochs 2   --lr 2e-4   --batch_size 1   --grad_accum 16   --lora_r 32   --lora_alpha 16   --lora_dropout 0.05
```

## 5. 推理测试（加载适配器）
推理（纯基座对比）：
```bash
python scripts/infer_qwen.py \
  --model ./models/Qwen2.5-7B-Instruct \
  --prompt "用两句话介绍沐霂这个Vtuber。" \
  --use_chat_template
```

叠加 LoRA 对比：
```bash
python scripts/infer_qwen.py \
  --model ./models/Qwen2.5-7B-Instruct \
  --peft_dir qwen25-vtuber-qlora \
  --prompt "用两句话介绍沐霂这个Vtuber。" \
  --use_chat_template
```

## 小贴士
- `deactivate` 可退出虚拟环境。
- 若 `bitsandbytes` 有兼容问题，可指定版本并更新依赖：  
  ```bash
  uv add bitsandbytes==0.43.1
  ```
- 训练时可用 `watch -n 2 nvidia-smi` 监控显存占用。

---

祝你训练顺利！
