import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ===== KONFIGURACJA =====
INDEX_FILE = Path("knowledge/faiss/rose_index.faiss")
META_FILE = Path("knowledge/faiss/rose_chunks_meta.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
# ========================


def main():
    if not INDEX_FILE.exists():
        raise FileNotFoundError("❌ Brak indexu FAISS")

    if not META_FILE.exists():
        raise FileNotFoundError("❌ Brak metadanych chunków")

    print("🔹 Ładowanie FAISS indexu...")
    index = faiss.read_index(str(INDEX_FILE))

    print("🔹 Ładowanie chunków...")
    chunks = json.loads(META_FILE.read_text(encoding="utf-8"))

    print("🔹 Ładowanie modelu embeddingów...")
    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\n❓ Pytanie (ENTER = wyjście): ").strip()
        if not query:
            break

        query_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = index.search(query_emb, TOP_K)

        print("\n🔍 WYNIKI:")
        for rank, idx in enumerate(indices[0], 1):
            chunk = chunks[idx]
            score = scores[0][rank - 1]

            print(f"\n--- #{rank} | score={score:.3f} | {chunk['id']} ---")
            print(chunk["text"][:800])
            print("...")

    print("\n✅ Test zakończony")


if __name__ == "__main__":
    main()
