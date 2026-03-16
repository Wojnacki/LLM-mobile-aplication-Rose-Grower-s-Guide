import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("knowledge/chunks/poradnik_pielegnacji_roz_chunks_fixed.jsonl")
INDEX_FILE  = Path("knowledge/faiss/rose_index.faiss")
META_FILE   = Path("knowledge/faiss/rose_chunks_meta.json")
MODEL_NAME  = "intfloat/multilingual-e5-small"

def load_chunks(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def chunk_to_passage(c):
    keywords = ', '.join(c.get('keywords', []))
    title    = c.get('title', '')
    text     = c.get('text', '')
    return f"passage: {title}. {text} Słowa kluczowe: {keywords}"

def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Brak pliku: {CHUNKS_FILE}")

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Ładowanie chunków...")
    chunks = load_chunks(CHUNKS_FILE)

    texts = [chunk_to_passage(c) for c in chunks]
    texts = [t for t in texts if len(t.strip()) > 20]  # walidacja

    print(f"Chunków: {len(texts)}")
    print(f"Przykład:\n{texts[0][:300]}\n")

    print("Ładowanie modelu...")
    model = SentenceTransformer(MODEL_NAME)

    print("Tworzenie embeddingów...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    )

    dim = embeddings.shape[1]
    print(f"Wymiar embeddingów: {dim}")

    print("Budowa indeksu FAISS...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))

    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Gotowe — {index.ntotal} wektorów w indeksie")

if __name__ == "__main__":
    main()