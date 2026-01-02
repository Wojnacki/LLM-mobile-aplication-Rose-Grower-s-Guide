import json
from pathlib import Path

INPUT = Path("knowledge/chunks/poradnik_pielegnacji_roz_chunks.jsonl")
OUTPUT = Path("knowledge/chunks/poradnik_pielegnacji_roz_chunks_fixed.jsonl")

MIN_CHARS = 200


def load_chunks(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_chunks(chunks, path):
    with path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, 1):
            c["id"] = f"rose_chunk_{i:03}"
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main():
    chunks = load_chunks(INPUT)

    fixed = []
    skip_next = False

    for i in range(len(chunks)):
        if skip_next:
            skip_next = False
            continue

        current = chunks[i]
        text = current["text"]

        if len(text) < MIN_CHARS and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]

            merged_text = text.rstrip() + "\n\n" + next_chunk["text"].lstrip()

            fixed.append({
                "text": merged_text
            })

            skip_next = True
        else:
            fixed.append({
                "text": text
            })

    save_chunks(fixed, OUTPUT)

    print(f"✅ Naprawiono chunki")
    print(f"📄 Zapisano do: {OUTPUT}")
    print(f"📦 Liczba chunków: {len(fixed)}")


if __name__ == "__main__":
    main()
