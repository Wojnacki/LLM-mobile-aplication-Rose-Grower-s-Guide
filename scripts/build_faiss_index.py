import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ===== KONFIGURACJA =====
CHUNKS_FILE = Path("knowledge/chunks/poradnik_pielegnacji_roz_chunks_fixed.jsonl")
INDEX_FILE = Path("knowledge/faiss/rose_index.faiss")
META_FILE = Path("knowledge/faiss/rose_chunks_meta.json")

MODEL_NAME = "intfloat/multilingual-e5-small"
#MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ========================


def load_chunks(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"❌ Brak pliku: {CHUNKS_FILE}")

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("🔹 Ładowanie chunków...")
    chunks = load_chunks(CHUNKS_FILE)
    texts = [
    f"{c.get('title','')} {c.get('section','')} {c.get('text','')} {' '.join(c.get('keywords', []))}"
    for c in chunks
    ]

    print("\n🔎 Przykładowy tekst do embeddingu:\n")
    print(texts[0][:400])

    print(f"🔹 Chunków: {len(texts)}")

    print("🔹 Ładowanie modelu embeddingów...")
    model = SentenceTransformer(MODEL_NAME)

    print("🔹 Tworzenie embeddingów...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    print(f"🔹 Wymiar embeddingów: {dim}")

    print("🔹 Budowa indeksu FAISS...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n✅ FAISS index gotowy")
    print(f"📦 Index: {INDEX_FILE}")
    print(f"📄 Meta:  {META_FILE}")


if __name__ == "__main__":
    main()
