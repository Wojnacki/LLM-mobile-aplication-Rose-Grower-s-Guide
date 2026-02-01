import json
import re
from pathlib import Path
import sys


# ===== KONFIG =====
CHUNKS_FILE = Path("knowledge/chunks/poradnik_pielegnacji_roz_chunks_fixed.jsonl")

TYPES = [
    "rabatowe",
    "parkowe",
    "pnące",
    "okrywowe",
    "pienne"
]

MIN_OCCURRENCES = 2
WINDOW_CHARS = 300
# ==================


def extract_header_and_body(text: str):
    lines = text.splitlines()
    header = lines[0].lower()
    body = "\n".join(lines[1:]).lower()
    return header, body


def count_occurrences(text: str, keyword: str):
    return len(re.findall(rf"\b{keyword}\b", text))


def main():
    errors = []

    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            text = chunk["text"]

            header, body = extract_header_and_body(text)
            body_head = body[:WINDOW_CHARS]

            for rose_type in TYPES:
                if rose_type in header:
                    count = count_occurrences(body_head, rose_type)
                    if count < MIN_OCCURRENCES:
                        errors.append({
                            "id": chunk["id"],
                            "type": rose_type,
                            "found": count
                        })

    if errors:
        print("\n❌ SEMANTIC ANCHOR VALIDATION FAILED\n")
        for e in errors:
            print(
                f"- {e['id']} | typ: {e['type']} | "
                f"wystąpienia w pierwszych {WINDOW_CHARS} znakach: {e['found']}"
            )
        sys.exit

    print("\n✅ Semantic anchor validation PASSED")

def new_func():
    return 1


if __name__ == "__main__":
    main()
