import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ===== KONFIG =====
INDEX_FILE = Path("knowledge/faiss/rose_index.faiss")
META_FILE = Path("knowledge/faiss/rose_chunks_meta.json")
QUESTIONS_FILE = Path("tests/rag_questions.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
# ==================


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def is_relevant(chunk_text: str, must_contain: list[str]) -> bool:
    text = chunk_text.lower()
    return all(token in text for token in must_contain)


def main():
    index = faiss.read_index(str(INDEX_FILE))
    chunks = load_json(META_FILE)
    questions = load_json(QUESTIONS_FILE)

    model = SentenceTransformer(MODEL_NAME)

    top1_hits = 0
    topk_hits = 0

    print("\n📊 TEST JAKOŚCI RAG\n")

    for q in questions:
        query = q["question"]
        must = q["must_contain"]

        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = index.search(q_emb, TOP_K)

        hits = []
        for idx in indices[0]:
            hits.append(chunks[idx]["text"])

        top1_ok = is_relevant(hits[0], must)
        topk_ok = any(is_relevant(h, must) for h in hits)

        top1_hits += int(top1_ok)
        topk_hits += int(topk_ok)

        print(f"❓ {query}")
        print(f"   top-1: {'✅' if top1_ok else '❌'}")
        print(f"   top-{TOP_K}: {'✅' if topk_ok else '❌'}")

    total = len(questions)

    print("\n📈 WYNIKI KOŃCOWE")
    print(f"Top-1 accuracy: {top1_hits}/{total} ({top1_hits/total:.0%})")
    print(f"Top-{TOP_K} accuracy: {topk_hits}/{total} ({topk_hits/total:.0%})")

    print("\n✅ Test zakończony")


if __name__ == "__main__":
    main()
