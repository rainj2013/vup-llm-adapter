#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/extract_with_llm.py  (Moonshot Kimi K2 版本，带节流+重试)

功能
- 读取 data/clean_json/*.json
- 调用大模型抽取结构化信息，输出到 data/extracted_json/*.extracted.json
- 可选 --gen_qa 追加问答到 outputs/vtuber_qa.from_llm.jsonl
- 断点续跑：每条完成即落盘；已有文件默认跳过（可 --overwrite）

提供方 & 模型
- moonshot (Kimi)：OpenAI 兼容 /v1/chat/completions，示例模型：
  kimi-k2-0905-preview、kimi-k2-0711-preview（以你账号可用为准）
- openai：保留作为备选
- dashscope（通义）：保留作为备选

环境变量
- Moonshot：
  MOONSHOT_API_KEY=sk-xxxx
  （可选）MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
- OpenAI：
  OPENAI_API_KEY=sk-xxxx
  （可选）OPENAI_BASE_URL=https://api.openai.com/v1
- DashScope：
  DASHSCOPE_API_KEY=sk-xxxx

用法示例（优先 moonshot）
  export MOONSHOT_API_KEY=sk-...
  python scripts/extract_with_llm.py --provider moonshot --model kimi-k2-0905-preview \
    --gen_qa --rpm 20 --max_retries 8

依赖：requests、tqdm
  uv add requests tqdm
"""

import os
import re
import json
import time
import random
import argparse
from urllib.parse import quote
from typing import Dict, Any, List, Optional

import requests
from tqdm import tqdm

SYSTEM_PROMPT = (
    "你是一个信息抽取助手，懂 Vtuber 设定与历史。"
    "从给定的维基页面内容中提取结构化信息。只输出 JSON，不要解释说明。"
    "字段包括：\n"
    "- name: Vtuber 名称\n"
    "- intro: 一句话简介，回答「xxx是谁？」（≤60字）\n"
    "- debut: 出道日期（如果有）\n"
    "- agency: 所属社团/经纪公司（如果有）\n"
    "- language: 使用语言（数组，若有）\n"
    "- platform: 活动平台（数组，若有）\n"
    "- fandom: 粉丝名（如果有）\n"
    "- color: 应援色（如果有）\n"
    "- origin: 出身/国籍（如果有）\n"
    "- birthday: 生日（如果有）\n"
    "要求：1) 只返回合法 JSON；2) 缺失字段返回 null 或空数组；3) 日期尽量规范。"
)
USER_PROMPT_TEMPLATE = "请根据以下文本抽取字段并输出 JSON：\n---\n【标题】{title}\n{body}\n---\n"

DEFAULT_MIN_INTRO_LEN = 10
DEFAULT_MAX_CHARS = 16000

UA = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"}

RETRIABLE_STATUS = {429, 500, 502, 503, 504}

# ----------------------------- Rate limiter -----------------------------
class RateLimiter:
    def __init__(self, rpm: int = 0):
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.last_ts = 0.0
    def wait(self):
        if self.min_interval <= 0: return
        now = time.time()
        delta = self.min_interval - (now - self.last_ts)
        if delta > 0: time.sleep(delta)
        self.last_ts = time.time()

# ----------------------------- Provider select -----------------------------
def find_provider(explicit: Optional[str] = None) -> str:
    if explicit: return explicit.lower()
    if os.getenv("MOONSHOT_API_KEY"): return "moonshot"
    if os.getenv("OPENAI_API_KEY"):   return "openai"
    if os.getenv("DASHSCOPE_API_KEY"):return "dashscope"
    raise RuntimeError("缺少凭据：请设置 MOONSHOT_API_KEY（推荐）或 OPENAI_API_KEY / DASHSCOPE_API_KEY。")

# ----------------------------- HTTP with retries -----------------------------
def _sleep_for_retry(resp: Optional[requests.Response], attempt: int, backoff_base: float, jitter: float):
    # 优先尊重 Retry-After
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                secs = float(ra)
                time.sleep(secs + random.uniform(0, jitter))
                return
            except Exception:
                pass
    # 指数退避 + 抖动
    delay = (backoff_base ** attempt) + random.uniform(0, jitter)
    time.sleep(delay)

def post_with_retries(url: str, headers: dict, payload: dict, timeout: float,
                      max_retries: int, backoff_base: float, jitter: float) -> requests.Response:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in RETRIABLE_STATUS:
                # 打印诊断信息
                try:
                    js = r.json()
                    print(f"[HTTP {r.status_code}] {js.get('error', js)}")
                except Exception:
                    print(f"[HTTP {r.status_code}] {r.text[:200]}")
                if attempt < max_retries:
                    _sleep_for_retry(r, attempt+1, backoff_base, jitter)
                    continue
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            body = None
            try: body = e.response.json()
            except Exception: body = e.response.text if e.response is not None else None
            print(f"[HTTPError] {e} | body={body} | headers={dict(e.response.headers) if e.response else {}}")
            if e.response is not None and e.response.status_code in RETRIABLE_STATUS and attempt < max_retries:
                _sleep_for_retry(e.response, attempt+1, backoff_base, jitter); last_err = e; continue
            raise
        except requests.RequestException as e:
            print(f"[ReqError] {e}")
            if attempt < max_retries:
                _sleep_for_retry(None, attempt+1, backoff_base, jitter); last_err = e; continue
            raise
    assert last_err is not None
    raise last_err

# ----------------------------- API callers -----------------------------
def call_moonshot_chat(model: str, system: str, user: str, timeout: float,
                       max_retries: int, backoff_base: float, jitter: float) -> str:
    """
    Moonshot/Kimi：OpenAI 兼容 chat completions
    默认基址：https://api.moonshot.cn/v1
    文档：/v1/chat/completions
    """
    base = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1").rstrip("/")
    url  = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('MOONSHOT_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":user},
        ],
        "temperature": 0.1,
        "top_p": 0.95,
    }
    r = post_with_retries(url, headers, payload, timeout, max_retries, backoff_base, jitter)
    js = r.json()
    return js["choices"][0]["message"]["content"]

def call_openai_chat(model: str, system: str, user: str, timeout: float,
                     max_retries: int, backoff_base: float, jitter: float) -> str:
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url  = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":user},
        ],
        "temperature": 0.1,
        "top_p": 0.95,
    }
    r = post_with_retries(url, headers, payload, timeout, max_retries, backoff_base, jitter)
    js = r.json()
    return js["choices"][0]["message"]["content"]

def call_dashscope_chat(model: str, system: str, user: str, timeout: float,
                        max_retries: int, backoff_base: float, jitter: float) -> str:
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": {"messages":[
            {"role":"system","content":system},
            {"role":"user","content":user},
        ]},
        "parameters": {"result_format":"message","temperature":0.1,"top_p":0.95}
    }
    r = post_with_retries(url, headers, payload, timeout, max_retries, backoff_base, jitter)
    js = r.json()
    try:
        return js["output"]["choices"][0]["message"]["content"]
    except Exception:
        return js.get("output", {}).get("text", "")

# ----------------------------- Utils -----------------------------
def read_clean_record(path: str) -> Dict[str, Any]:
    obj = json.load(open(path, encoding="utf-8"))
    title = obj.get("title") or os.path.basename(path).replace(".json", "")
    info = obj.get("info") or {}
    text = obj.get("text") or ""
    info_prefix = ""
    if info:
        items = list(info.items())[:10]
        kv = "\n".join(f"{k}: {v}" for k, v in items if v)
        info_prefix = f"【信息框（节选）】\n{kv}\n\n"
    return {"title": title, "body": info_prefix + text}

def normalize_json_fields(x: Dict[str, Any]) -> Dict[str, Any]:
    def as_list(v):
        if v is None: return []
        if isinstance(v, list): return [str(i).strip() for i in v if str(i).strip()]
        return [str(v).strip()] if str(v).strip() else []
    out = {
        "name":     x.get("name"),
        "intro":    x.get("intro"),
        "debut":    x.get("debut"),
        "agency":   x.get("agency"),
        "language": as_list(x.get("language")),
        "platform": as_list(x.get("platform")),
        "fandom":   x.get("fandom"),
        "color":    x.get("color"),
        "origin":   x.get("origin"),
        "birthday": x.get("birthday"),
    }
    for k,v in list(out.items()):
        if isinstance(v, str) and not v.strip():
            out[k] = None
    return out

def parse_llm_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S|re.I)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    raise ValueError("无法从模型输出中解析 JSON。")

def make_qa_samples(rec: Dict[str, Any], site_base: str) -> List[Dict[str, Any]]:
    title = rec.get("name") or rec.get("title")
    url = site_base.rstrip("/") + "/" + quote(str(title), safe="/()")
    sys_msg = "你是懂Vtuber设定与历史的助理。回答要客观、简洁。"
    def msg(q, a):
        return {"messages":[
            {"role":"system","content":sys_msg},
            {"role":"user","content":f"【问】{q}"},
            {"role":"assistant","content":f"【答】{a}。来源：{url}"},
        ]}
    out=[]
    intro = rec.get("intro")
    if intro and len(str(intro)) >= DEFAULT_MIN_INTRO_LEN:
        out.append(msg(f"{title}是谁？", str(intro).strip()))
    fields = [
        ("debut","出道日期"),
        ("agency","所属社"),
        ("language","使用语言"),
        ("platform","活动平台"),
        ("fandom","粉丝名"),
        ("color","应援色"),
        ("origin","出身/国籍"),
        ("birthday","生日"),
    ]
    for key,label in fields:
        val = rec.get(key)
        if isinstance(val, list):
            val = "、".join([str(i) for i in val if str(i).strip()])
        if val and str(val).strip():
            out.append(msg(f"{title}的{label}是什么？", f"{label}：{val}"))
    return out

def truncate_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars: return s
    cut = s[:max_chars]
    p = cut.rfind("\n")
    return cut if p < 4000 else cut[:p]

# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/clean_json")
    ap.add_argument("--outdir", default="data/extracted_json")
    ap.add_argument("--qa_out", default="outputs/vtuber_qa.from_llm.jsonl")
    ap.add_argument("--gen_qa", action="store_true")
    ap.add_argument("--site_base", default="https://zh.moegirl.org.cn")
    ap.add_argument("--provider", choices=["moonshot","openai","dashscope"], default=None)
    ap.add_argument("--model", default="kimi-k2-0905-preview")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timeout", type=float, default=60.0)
    # 速率/重试
    ap.add_argument("--rpm", type=int, default=0, help="每分钟请求上限（0=不限制）")
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--backoff_base", type=float, default=1.6)
    ap.add_argument("--jitter", type=float, default=0.6)
    args = ap.parse_args()

    provider = find_provider(args.provider)
    os.makedirs(args.outdir, exist_ok=True)
    if args.gen_qa: os.makedirs(os.path.dirname(args.qa_out), exist_ok=True)

    files = [f for f in os.listdir(args.indir) if f.endswith(".json")]
    files.sort()
    if args.limit > 0: files = files[:args.limit]

    limiter = RateLimiter(args.rpm)

    wrote_qa = 0; skipped = 0; ok = 0; fail = 0
    qa_fh = open(args.qa_out, "a", encoding="utf-8") if args.gen_qa else None

    for fn in tqdm(files, desc="LLM Extract"):
        in_path = os.path.join(args.indir, fn)
        title = fn[:-5]
        out_path = os.path.join(args.outdir, f"{title}.extracted.json")

        if (not args.overwrite) and os.path.exists(out_path):
            print(f"[SKIP] {title} -> 已存在 {out_path}")
            skipped += 1
            continue

        try:
            rec = read_clean_record(in_path)
            title = rec["title"] or title
            body  = truncate_text(rec["body"] or "", args.max_chars)
            user_prompt = USER_PROMPT_TEMPLATE.format(title=title, body=body)

            # 节流
            limiter.wait()

            # 调用对应提供方
            if provider == "moonshot":
                resp_text = call_moonshot_chat(
                    args.model, SYSTEM_PROMPT, user_prompt,
                    timeout=args.timeout, max_retries=args.max_retries,
                    backoff_base=args.backoff_base, jitter=args.jitter
                )
            elif provider == "openai":
                resp_text = call_openai_chat(
                    args.model, SYSTEM_PROMPT, user_prompt,
                    timeout=args.timeout, max_retries=args.max_retries,
                    backoff_base=args.backoff_base, jitter=args.jitter
                )
            else:
                resp_text = call_dashscope_chat(
                    args.model, SYSTEM_PROMPT, user_prompt,
                    timeout=args.timeout, max_retries=args.max_retries,
                    backoff_base=args.backoff_base, jitter=args.jitter
                )

            raw = parse_llm_json(resp_text)
            extracted = normalize_json_fields(raw)
            if not extracted.get("name"): extracted["name"] = title
            if extracted.get("intro") and len(str(extracted["intro"])) < DEFAULT_MIN_INTRO_LEN:
                extracted["intro"] = None

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(extracted, f, ensure_ascii=False, indent=2)

            if qa_fh and args.gen_qa:
                for s in make_qa_samples(extracted, args.site_base):
                    qa_fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                    wrote_qa += 1

            ok += 1

        except Exception as e:
            print(f"[ERROR] {title}: {e}")
            fail += 1

    if qa_fh: qa_fh.close()

    print(f"[DONE] provider={provider} model={args.model} | ok={ok} skip={skipped} fail={fail} "
          f"| qa_wrote={wrote_qa} -> {args.qa_out if args.gen_qa else 'N/A'}")

if __name__ == "__main__":
    main()

