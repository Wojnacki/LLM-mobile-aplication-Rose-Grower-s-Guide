import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_MD = BASE_DIR / "knowledge" / "raw" / "poradnik_pielegnacji_roz_v_7.md"
OUTPUT_JSONL = BASE_DIR / "knowledge" / "chunks" / "poradnik_pielegnacji_roz_chunks.jsonl"

MAX_CHARS = 2000


def normalize(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def split_by_headers(text: str):
    pattern = r"(?=^## |\n## |\n### )"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    return [normalize(p) for p in parts if p.strip()]


def split_large_chunk(chunk: str):
    if len(chunk) <= MAX_CHARS:
        return [chunk]

    paragraphs = chunk.split("\n\n")
    result, current = [], ""

    for p in paragraphs:
        if len(current) + len(p) <= MAX_CHARS:
            current += ("\n\n" + p if current else p)
        else:
            result.append(current)
            current = p

    if current:
        result.append(current)

    return result


def main():
    if not INPUT_MD.exists():
        raise FileNotFoundError(f"❌ Nie znaleziono pliku: {INPUT_MD}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    md_text = INPUT_MD.read_text(encoding="utf-8")
    raw_chunks = split_by_headers(md_text)

    chunk_id = 1
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for raw in raw_chunks:
            for part in split_large_chunk(raw):
                f.write(json.dumps({
                    "id": f"rose_chunk_{chunk_id:03}",
                    "text": part
                }, ensure_ascii=False) + "\n")
                chunk_id += 1

    print(f"✅ Utworzono {chunk_id - 1} chunków")
    print(f"📄 Plik: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
