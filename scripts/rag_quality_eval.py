import json
import logging
import numpy as np
from pathlib import Path
import faiss
import onnxruntime as ort
from transformers import AutoTokenizer

# Wycisz fałszywy warning o regex tokenizera
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

INDEX_FILE         = Path("mobile/flutter_app/assets/rose_index.faiss")
META_FILE          = Path("mobile/flutter_app/assets/rose_chunks_meta.json")
QUESTIONS_FILE     = Path("tests/rag_questions.json")
MODEL_DIR          = Path("models/onnx/e5-small-int8")
TOP_K              = 5
MIN_SCORE          = 0.50
MIN_SCORE_NEGATIVE = 0.85

# ── ONNX helpers ────────────────────────────────────────────────────────────

def load_model(model_dir: Path):
    tokenizer   = AutoTokenizer.from_pretrained(str(model_dir))
    sess        = ort.InferenceSession(
        str(model_dir / "model.onnx"),
        providers=["CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in sess.get_inputs()]
    return tokenizer, sess, input_names

def encode_query(query: str, tokenizer, sess, input_names) -> np.ndarray:
    """Enkoduje pojedyncze zapytanie z prefixem query: i zwraca znormalizowany wektor."""
    encoded = tokenizer(
        f"query: {query}",
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512
    )
    feed = {k: v for k, v in encoded.items() if k in input_names}
    if "token_type_ids" in input_names and "token_type_ids" not in feed:
        feed["token_type_ids"] = np.zeros_like(encoded["input_ids"])

    outputs = sess.run(None, feed)

    # Mean pooling
    token_embeddings = outputs[0]
    mask             = encoded["attention_mask"][..., np.newaxis].astype(np.float32)
    pooled           = (token_embeddings * mask).sum(1) / mask.sum(1).clip(min=1e-9)

    # Normalizacja
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / norm.clip(min=1e-9)).astype(np.float32)

# ── Ewaluacja ────────────────────────────────────────────────────────────────

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def is_relevant(chunk: dict, must_contain: list[str]) -> bool:
    haystack = " ".join([
        chunk.get("text", ""),
        chunk.get("title", ""),
        " ".join(chunk.get("keywords", []))
    ]).lower()
    return all(token[:5] in haystack for token in must_contain)

def main():
    print("Ładowanie indeksu i modelu ONNX...")
    index     = faiss.read_index(str(INDEX_FILE))
    chunks    = load_json(META_FILE)
    questions = load_json(QUESTIONS_FILE)
    tokenizer, sess, input_names = load_model(MODEL_DIR)
    print(f"Model: {MODEL_DIR.name} | Wektorów w indeksie: {index.ntotal}\n")

    top1_hits = 0
    topk_hits = 0
    neg_total = 0
    neg_hits  = 0

    print("📊 TEST JAKOŚCI RAG\n")

    for q in questions:
        query           = q["question"]
        must            = q.get("must_contain", [])
        expect_negative = q.get("expect_no_results", False)

        q_emb = encode_query(query, tokenizer, sess, input_names)
        scores, indices = index.search(q_emb, TOP_K)

        hits = [
            (chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if score >= MIN_SCORE
        ]

        print(f"❓ {query}")

        # ── TEST NEGATYWNY ───────────────────────────────────────────────
        if expect_negative:
            neg_total += 1
            max_score = hits[0][1] if hits else 0.0
            passed    = max_score < MIN_SCORE_NEGATIVE
            neg_hits += int(passed)

            if passed:
                print(f"   [NEG] max_score={max_score:.3f} — ✅ poprawnie odrzucono (próg={MIN_SCORE_NEGATIVE})")
            else:
                print(f"   [NEG] max_score={max_score:.3f} — ❌ fałszywy pozytyw (powyżej progu {MIN_SCORE_NEGATIVE})")
                for rank, (chunk, score) in enumerate(hits):
                    print(f"         [{rank+1}] score={score:.3f} | {chunk['title']}")
            print()
            continue

        # ── TEST POZYTYWNY ───────────────────────────────────────────────
        for rank, (chunk, score) in enumerate(hits):
            relevant = is_relevant(chunk, must)
            print(f"   [{rank+1}] score={score:.3f} | {chunk['title']} {'✅' if relevant else '❌'}")

        top1_ok = bool(hits) and is_relevant(hits[0][0], must)
        topk_ok = any(is_relevant(h, must) for h, _ in hits)

        top1_hits += int(top1_ok)
        topk_hits += int(topk_ok)
        print()

    # ── PODSUMOWANIE ─────────────────────────────────────────────────────
    pos_total = len(questions) - neg_total
    print("📈 WYNIKI KOŃCOWE")
    print(f"Pytania pozytywne  — Top-1 accuracy : {top1_hits}/{pos_total} ({top1_hits/pos_total:.0%})")
    print(f"Pytania pozytywne  — Top-{TOP_K} accuracy : {topk_hits}/{pos_total} ({topk_hits/pos_total:.0%})")
    if neg_total:
        print(f"Pytania negatywne  — odrzucono poprawnie: {neg_hits}/{neg_total} ({neg_hits/neg_total:.0%})  [próg={MIN_SCORE_NEGATIVE}]")

    # Test pojedynczego pytania
    query = 'jakie są rodzaje róż'
    q_emb = encode_query(query, tokenizer, sess, input_names)
    scores, indices = index.search(q_emb, 5)

    print(f'\nTest pojedynczego pytania: {query}')
    for rank, idx in enumerate(indices[0], 1):
        chunk = chunks[idx]
        score = scores[0][rank - 1]
        print(f'[{rank}] score={score:.3f} | {chunk["title"]}')
        print(f'   Text: {chunk["text"][:200]}...')
        print()
