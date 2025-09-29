#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 data/clean_json/*.json 生成 Qwen 聊天式 JSONL

策略：
- 先用清洗后的 text 选第一段简介（仅做净化与长度判断）
- 若简介过短/缺失，调用 MediaWiki action=parse 取渲染后的 <p> 文本兜底
- 信息框键名做归一化 + 正则匹配，并提供正文兜底抽取
- 每个条目至少力争产出两类问答： '是谁？' + 若干信息框/正文兜底问答
- 提供 --debug 输出，统计产出/兜底/过滤原因
"""

import argparse
import os
import json
import re
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

SYSTEM_PROMPT = "你是懂Vtuber设定与历史的助理。回答要客观、简洁。"
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}

# ---------- 文本清洗与工具 ----------

def sanitize(s: str) -> str:
    """温和去噪：ref/HTML/模板花括号/管道/多余空白/重复标点。"""
    if not s:
        return ""
    s = re.sub(r"<ref.*?</ref>|<ref.*?/>", "", s, flags=re.S | re.I)
    s = re.sub(r"<.*?>", "", s)
    s = re.sub(r"[{}]+", "", s)
    s = re.sub(r"\s*\|\s*", " ", s)
    s = re.sub(r"[ \t\r]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"([。！？!?])\1{1,}", r"\1", s)
    return s.strip()


def pick_intro_from_text(text: str, min_len: int) -> str:
    """从本地 clean 文本中挑第一段够长的简介；否则退化到截断+裁句。"""
    if not text:
        return ""
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        p = sanitize(para)
        if len(p) >= min_len:
            return p
    t = sanitize(text)[:240]
    m = re.search(r"[。！？!?]", t)
    return t[:m.end()] if m else t


def fetch_rendered_intro(api_base: str, title: str, min_len: int) -> str:
    """用 action=parse 拉渲染后的 HTML，再取首个够长的 <p> 兜底。"""
    try:
        params = {"action": "parse", "page": title, "prop": "text",
                  "format": "json", "formatversion": 2}
        r = requests.get(api_base, params=params, headers=UA, timeout=60)
        r.raise_for_status()
        js = r.json()
        html = js.get("parse", {}).get("text", "")
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for p in soup.select("p"):
            txt = sanitize(p.get_text(" ", strip=True))
            if len(txt) >= min_len:
                return txt
        return ""
    except Exception:
        return ""


def normalize_key(s: str) -> str:
    """键名归一化：全角->半角、去空白、小写。"""
    if not s:
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def split_values(s: str):
    """将多值字段按常见分隔符拆分并去重。"""
    parts = re.split(r"[\/;；、|，,]+", s)
    parts = [p.strip() for p in parts if p.strip()]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ---------- 信息框映射与兜底 ----------

INFO_MAP = {
    "出道日期":  [r"出道|出道日期|debut|首次登场|初配信|初登場|初登场"],
    "所属社":    [r"经纪公司|所(?:属|屬)|公司|agency|事务所|事务(?:所)?|团体|團體|所属组别|社|企划|企劃|project"],
    "语言":      [r"语(?:言)?|lang(uage)?|語言|使用語言|使用语言|主要语言"],
    "平台":      [r"平台|platform|活动平台|主要平台|直播平台|b站|bilibili|youtube|twitch"],
    "粉丝名":    [r"粉丝名|粉丝称呼|fan\s*name|应援名|粉丝昵称|粉丝称谓|粉絲名|粉絲稱呼"],
    "应援色":    [r"应援色|应援顏色|应援色彩|应援颜色|color|代表色|形象色"],
    "出身/国籍":[r"国籍|出身|国家|region|地区|籍贯|出生地|國籍"],
    "生日":      [r"生日|生辰|birthday|出生日期|诞生日|诞生(?:日|月日)"],
}

FB_PATS_IN_TEXT = {
    "粉丝名":  r"(粉丝名|粉丝称呼|粉絲名|fan\s*name)[：:]\s*([^\n。；;，,]{2,30})",
    "应援色":  r"(应援色|代表色|形象色)[：:]\s*([^\n。；;，,]{1,20})",
    "生日":    r"(生日|诞生日|出生日期)[：:]\s*([0-9０-９]{2,4}[^。；;，,\n]{0,12})",
    "所属社":  r"(所属|经纪公司|公司|事务所)[：:]\s*([^\n。；;，,]{2,40})",
    "语言":    r"(语言|語言|language)[：:]\s*([^\n。；;，,]{1,30})",
    "平台":    r"(平台|直播平台|活动平台)[：:]\s*([^\n。；;，,]{1,40})",
}


def map_info_qas(title: str, info: dict, text_for_fallback: str = "", want_labels=None, debug=False):
    """从信息框与正文兜底产问答。"""
    if want_labels is None:
        want_labels = list(INFO_MAP.keys())
    qas = []
    info = info or {}

    # 键名归一化映射
    norm_info = {}
    for k, v in info.items():
        nk = normalize_key(k)
        if nk and v:
            norm_info[nk] = (k, sanitize(str(v)))

    for label in want_labels:
        pats = INFO_MAP.get(label, [])
        value = None
        matched_key = None

        # 1) 信息框键匹配
        for nk, (ok, vv) in norm_info.items():
            if value:
                break
            for pat in pats:
                if re.search(pat, nk, flags=re.I):
                    if vv and len(vv) >= 2:
                        value = vv
                        matched_key = ok
                        break

        # 2) 正文兜底（信息框无命中时）
        if not value and text_for_fallback and label in FB_PATS_IN_TEXT:
            m = re.search(FB_PATS_IN_TEXT[label], text_for_fallback, flags=re.I)
            if m:
                value = sanitize(m.group(2))

        if value:
            vals = split_values(value)
            if len(vals) > 3:
                vals = vals[:3]
            ans = "、".join(vals)
            qas.append((
                f"{title}的{label}是什么？",
                f"{label}：{ans}",
                matched_key or ""
            ))
        elif debug:
            print(f"[DEBUG] info-miss: {title} -> {label}")

    return qas


# ---------- 例子构建 ----------

def make_examples(rec: dict, api_base: str, site_base: str, min_len: int,
                  allow_parse_fallback: bool, include_info_qas: bool, debug: bool):
    title = rec.get("title") or ""
    text  = rec.get("text") or ""
    info  = rec.get("info") or {}

    # 1) 简介（先本地，再兜底 parse）
    intro = pick_intro_from_text(text, min_len=min_len)
    used_fallback = False
    if len(intro) < min_len and allow_parse_fallback and title:
        intro2 = fetch_rendered_intro(api_base, title, min_len=min_len)
        if len(intro2) >= min_len:
            intro = intro2
            used_fallback = True

    examples = []
    url = site_base.rstrip("/") + "/" + quote(title, safe="/()")

    if len(intro) >= min_len:
        q = f"{title}是谁？"
        a = sanitize(intro)
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "【问】" + q},
                {"role": "assistant", "content": "【答】" + a + f"。来源：{url}"}
            ]
        })

    # 2) 信息框/正文兜底问答
    if include_info_qas:
        for q, a, matched_key in map_info_qas(title, info, text_for_fallback=text, debug=debug):
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "【问】" + q},
                    {"role": "assistant", "content": "【答】" + sanitize(a) + f"。来源：{url}"}
                ]
            })
            if debug and matched_key:
                print(f"[DEBUG] info-hit: {title} '{matched_key}' -> {q}")

    if debug:
        reason = "ok" if examples else f"no_intro(len={len(intro)})"
        if used_fallback:
            reason += "|fallback_html"
        print(f"[DEBUG] {title}: {reason} | info_keys={len(info or {})}")

    return examples, used_fallback, len(intro)


# ---------- 主程序 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/clean_json", help="结构化 JSON 输入目录")
    ap.add_argument("--outfile", default="outputs/vtuber_qa.jsonl", help="输出 JSONL")
    ap.add_argument("--api_base", default="https://zh.moegirl.org.cn/api.php", help="MediaWiki API")
    ap.add_argument("--site_base", default="https://zh.moegirl.org.cn", help="页面前缀用于来源链接")
    ap.add_argument("--limit", type=int, default=0, help="最多处理文件数（0=全部）")
    ap.add_argument("--min_len", type=int, default=20, help="简介最短长度")
    ap.add_argument("--no_parse_fallback", action="store_true", help="禁用 action=parse 兜底")
    ap.add_argument("--no_info_qas", action="store_true", help="不生成信息框相关问答")
    ap.add_argument("--debug", action="store_true", help="打印调试信息")
    args = ap.parse_args()

    files = [f for f in os.listdir(args.indir) if f.endswith(".json")]
    files.sort()
    if args.limit > 0:
        files = files[:args.limit]

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    total = len(files)
    wrote = 0
    empty = 0
    used_parse = 0
    too_short = 0

    with open(args.outfile, "w", encoding="utf-8") as out:
        for fn in tqdm(files, desc="Build QA"):
            rec = json.load(open(os.path.join(args.indir, fn), encoding="utf-8"))
            examples, fallback_used, intro_len = make_examples(
                rec,
                api_base=args.api_base,
                site_base=args.site_base,
                min_len=args.min_len,
                allow_parse_fallback=not args.no_parse_fallback,
                include_info_qas=not args.no_info_qas,
                debug=args.debug,
            )
            if not examples:
                empty += 1
                if intro_len < args.min_len:
                    too_short += 1
            else:
                for ex in examples:
                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    wrote += 1
            if fallback_used:
                used_parse += 1

    print(f"[DONE] wrote samples: {wrote} | files: {total} | empty: {empty} "
          f"| intro<min: {too_short} | parse_fallback_used: {used_parse} "
          f"| out: {args.outfile}")


if __name__ == "__main__":
    main()

