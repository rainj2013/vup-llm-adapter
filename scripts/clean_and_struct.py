#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清洗：把 .wikitext 解析为 {title, info, text}
改动要点：
- 用 mwparserfromhell.strip_code() 移除模板/链接/表格，避免 "{{}}", "|" 残留
- 信息框字段同样 strip_code，一并去掉 ref/html
"""
import argparse, os, json, re
import mwparserfromhell
from tqdm import tqdm

def mw_clean(s: str) -> str:
    if not s:
        return ""
    code = mwparserfromhell.parse(s)
    txt = code.strip_code(normalize=True, collapse=True)
    # 去掉 ref / html tag 等
    txt = re.sub(r"<ref.*?</ref>|<ref.*?/>", "", txt, flags=re.S|re.I)
    txt = re.sub(r"<.*?>", "", txt)
    # 清掉可能漏网的花括号/管道/多余分隔
    txt = re.sub(r"[{}]+", "", txt)
    txt = re.sub(r"\s*\|\s*", " ", txt)
    # 合并空白
    txt = re.sub(r"[ \t\r]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def parse_infobox(wt: str):
    info = {}
    code = mwparserfromhell.parse(wt)
    for t in code.filter_templates():
        name = str(t.name).strip().lower()
        # 名称里出现 infobox/信息框/资料 等都视作信息框
        if any(k in name for k in ["infobox", "信息框", "资料"]):
            for p in t.params:
                k = str(p.name).strip()
                v = mw_clean(str(p.value))
                if v:
                    info[k] = v
    return info

def extract_clean_text(wt: str) -> str:
    return mw_clean(wt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/raw_pages", help="输入 .wikitext 目录")
    ap.add_argument("--outdir", default="data/clean_json", help="输出结构化 JSON 目录")
    ap.add_argument("--limit", type=int, default=0, help="最多处理文件数（0=全部）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = [f for f in os.listdir(args.indir) if f.endswith(".wikitext")]
    files.sort()
    if args.limit > 0:
        files = files[:args.limit]

    for fn in tqdm(files):
        wt = open(os.path.join(args.indir, fn), encoding="utf-8").read()
        title = fn[:-9]  # 去掉 .wikitext
        info = parse_infobox(wt)
        text = extract_clean_text(wt)
        rec = {"title": title, "info": info, "text": text}
        with open(os.path.join(args.outdir, fn.replace(".wikitext", ".json")), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    print("[DONE] Clean JSON saved to", args.outdir)

if __name__ == "__main__":
    main()

