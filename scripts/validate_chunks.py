import json
from pathlib import Path
from collections import Counter

# ====== KONFIGURACJA ======
CHUNKS_FILE = Path("knowledge/chunks/poradnik_pielegnacji_roz_chunks_fixed.jsonl")

MIN_CHARS = 200      # poniżej → podejrzane
MAX_CHARS = 2000     # powyżej → za duże
# ==========================


def load_chunks(path: Path):
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"❌ Brak pliku: {CHUNKS_FILE}")

    chunks = load_chunks(CHUNKS_FILE)

    sizes = [len(c["text"]) for c in chunks]

    print("\n📊 PODSTAWOWE STATYSTYKI")
    print(f"Liczba chunków: {len(chunks)}")
    print(f"Min znaków:     {min(sizes)}")
    print(f"Max znaków:     {max(sizes)}")
    print(f"Średnia:        {sum(sizes) // len(sizes)}")

    print("\n⚠️ POTENCJALNE PROBLEMY")

    small = [c for c in chunks if len(c["text"]) < MIN_CHARS]
    large = [c for c in chunks if len(c["text"]) > MAX_CHARS]

    print(f"Za małe (<{MIN_CHARS}): {len(small)}")
    print(f"Za duże (>{MAX_CHARS}): {len(large)}")

    if small:
        print("\n🔎 PRZYKŁAD MAŁEGO CHUNKU:")
        print(small[0]["id"])
        print(small[0]["text"][:300], "...")

    if large:
        print("\n🔎 PRZYKŁAD DUŻEGO CHUNKU:")
        print(large[0]["id"])
        print(large[0]["text"][:300], "...")

    print("\n🔁 DUPLIKATY")
    texts = [c["text"] for c in chunks]
    duplicates = [t for t, count in Counter(texts).items() if count > 1]
    print(f"Duplikaty: {len(duplicates)}")

    print("\n📌 NAGŁÓWKI")
    no_header = [c for c in chunks if not c["text"].lstrip().startswith("#")]
    print(f"Chunki bez nagłówka: {len(no_header)}")

    if no_header:
        print("\n🔎 PRZYKŁAD CHUNKU BEZ NAGŁÓWKA:")
        print(no_header[0]["id"])
        print(no_header[0]["text"][:300], "...")

    print("\n✅ WALIDACJA ZAKOŃCZONA")


if __name__ == "__main__":
    main()
