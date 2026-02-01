# scripts/auto_chunk_and_tag.py
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR))

import json
import re
from tag_rules import TYPE_RULES, TASK_RULES
from utils_md import split_by_headers


INPUT_MD = "knowledge/cleaned/poradnik_pielegnacji_roz_v_7.md"
OUTPUT_JSONL = "knowledge/chunks/poradnik_chunks.jsonl"


def detect_tag(text, rules, default="all"):
    text = text.lower()
    for tag, keywords in rules.items():
        for kw in keywords:
            if kw in text:
                return tag
    return default


def main():
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    chunks = split_by_headers(md_text)
    out = []

    for i, (header, body) in enumerate(chunks, 1):
        anchor_text = f"{header}\n{body[:300]}"

        type_tag = detect_tag(anchor_text, TYPE_RULES)
        task_tag = detect_tag(anchor_text, TASK_RULES)

        out.append({
            "id": f"rose_chunk_{i:03d}",
            "type": type_tag,
            "task": task_tag,
            "text": f"{header}\n\n{body}".strip()
        })

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Utworzono {len(out)} chunków → {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
