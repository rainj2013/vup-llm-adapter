#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch Vtuber-related pages from Moegirl's MediaWiki API (category-based).
Defaults to Chinese site & Category:虚拟YouTuber. Adjust via CLI.
"""
import argparse, os, time, json, sys
import urllib3
from urllib.parse import quote
import requests
from tqdm import tqdm

def category_members(base, category, limit=500, sleep=0.2, max_retries=5):
    # Ensure the category has the proper prefix
    if not category.startswith("Category:"):
        category = "Category:" + category
    
    cmc = None
    titles = []
    retries = 0
    while retries < max_retries:
        try:
            while True:
                params = {
                    "action": "query", "list": "categorymembers", "cmtitle": category,
                    "cmlimit": limit, "format": "json"
                }
                if cmc:
                    params["cmcontinue"] = cmc
                
                print(f"[INFO] Requesting category: {category}")  # Debug log
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
                }

                # Send the GET request with headers
                r = requests.get(base, params=params, headers=headers, timeout=60)
                r.raise_for_status()
                js = r.json()
                
                # Collect titles from the query result
                if "query" in js:
                    for it in js["query"]["categorymembers"]:
                        titles.append(it["title"])

                # Check if there is a next page (pagination)
                cont = js.get("continue", {})
                cmc = cont.get("cmcontinue")
                if not cmc:
                    break  # No more pages
                time.sleep(sleep)
            
            break  # If successful, break the retry loop
        except (requests.exceptions.RequestException, urllib3.exceptions.ProtocolError) as e:
            retries += 1
            print(f"[WARN] 请求失败 ({retries}/{max_retries})，正在重试: {e}")
            time.sleep(sleep * 2)  # Increase sleep before retrying
    if retries == max_retries:
        print("[ERROR] 达到最大重试次数，停止抓取。")
    return titles

def get_wikitext(base, title):
    params = {
        "action": "query", "prop": "revisions", "rvprop": "content",
        "rvslots": "main", "titles": title, "format": "json"
    }
    print(f"[INFO] Fetching wikitext for: {title}")  # Debug log

    # Set User-Agent header to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    # Send the GET request with headers
    r = requests.get(base, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    js = r.json()
    pages = js["query"]["pages"]
    page = next(iter(pages.values()))
    revs = page.get("revisions", [])
    if not revs: 
        return None
    return revs[0]["slots"]["main"]["*"]

def save_wikitext(title, wikitext, outdir):
    fn = os.path.join(outdir, title.replace("/", "_") + ".wikitext")
    with open(fn, "w", encoding="utf-8") as f:
        f.write(wikitext)
    print(f"[INFO] Saved wikitext for {title} to {fn}")

def load_processed_titles(processed_file):
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()

def save_processed_title(processed_file, title):
    with open(processed_file, "a", encoding="utf-8") as f:
        f.write(title + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://zh.moegirl.org.cn/api.php", help="MediaWiki API endpoint")
    ap.add_argument("--category", default="Category:在bilibili活动过的虚拟UP主", help="Category title to crawl")
    ap.add_argument("--outdir", default="data/raw_pages", help="Directory to save .wikitext files")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between API calls")
    ap.add_argument("--max", type=int, default=0, help="Max pages to fetch (0 = no limit)")
    ap.add_argument("--resume", action="store_true", help="Skip files that already exist")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load previously processed titles from file
    processed_titles_file = "processed_titles.txt"
    processed_titles = load_processed_titles(processed_titles_file)

    print(f"[INFO] Listing category members from: {args.category}")
    titles = category_members(args.base, args.category, limit=500, sleep=args.sleep)
    if args.max > 0:
        titles = titles[:args.max]

    print(f"[INFO] Fetched {len(titles)} titles. Downloading wikitext...")
    for t in tqdm(titles):
        # Check if already processed
        if t in processed_titles:
            print(f"[INFO] Skipping {t}, already processed.")  # Log for skipping
            continue
        
        # Check if file exists locally
        if args.resume and os.path.exists(os.path.join(args.outdir, t.replace("/", "_") + ".wikitext")):
            print(f"[INFO] Skipping {t}, file already exists.")  # Log for skipping
            save_processed_title(processed_titles_file, t)  # Even though skipped, mark it processed
            continue
        
        try:
            # If not processed, fetch and save
            print(f"[INFO] Fetching data for {t}...")  # Log for fetching
            wt = get_wikitext(args.base, t)
            if wt:
                save_wikitext(t, wt, args.outdir)
                # Mark title as processed after saving
                save_processed_title(processed_titles_file, t)
            time.sleep(args.sleep)
        except Exception as e:
            print(f"[WARN] Failed: {t}: {e}", file=sys.stderr)

    print("[DONE] Wikitext saved to", args.outdir)

if __name__ == "__main__":
    main()

