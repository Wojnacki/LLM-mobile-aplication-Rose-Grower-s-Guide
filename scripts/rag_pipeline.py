import json
import logging
import numpy as np
from pathlib import Path
import sys

import faiss
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import anthropic

# Add parent directory to path to import inference module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.config import ANTHROPIC_API_KEY

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# ── KONFIGURACJA ─────────────────────────────────────────────────────────────

INDEX_FILE  = Path("knowledge/faiss/rose_index.faiss")
META_FILE   = Path("knowledge/faiss/rose_chunks_meta.json")
MODEL_DIR   = Path("models/onnx/e5-small-int8")

CLAUDE_MODEL      = "claude-haiku-4-5-20251001"

TOP_K      = 3      # ile chunków podajemy do LLM
MIN_SCORE  = 0.75   # próg pewności — poniżej = "nie wiem"

# ── ONNX — enkodowanie zapytań ───────────────────────────────────────────────

def load_embedding_model(model_dir: Path):
    tokenizer   = AutoTokenizer.from_pretrained(str(model_dir))
    sess        = ort.InferenceSession(
        str(model_dir / "model.onnx"),
        providers=["CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in sess.get_inputs()]
    return tokenizer, sess, input_names

def encode_query(query: str, tokenizer, sess, input_names) -> np.ndarray:
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
    mask    = encoded["attention_mask"][..., np.newaxis].astype(np.float32)
    pooled  = (outputs[0] * mask).sum(1) / mask.sum(1).clip(min=1e-9)
    norm    = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / norm.clip(min=1e-9)).astype(np.float32)

# ── RETRIEVAL ────────────────────────────────────────────────────────────────

def retrieve(query: str, index, chunks: list, tokenizer, sess, input_names) -> list[dict]:
    q_emb             = encode_query(query, tokenizer, sess, input_names)
    scores, indices   = index.search(q_emb, TOP_K)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if score >= MIN_SCORE:
            results.append({
                "chunk": chunks[idx],
                "score": float(score)
            })
    return results

# ── PROMPT ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Jesteś ekspertem ogrodniczym specjalizującym się w uprawie róż.
Odpowiadasz wyłącznie na podstawie dostarczonego kontekstu.
Jeśli odpowiedź nie wynika z kontekstu, powiedz: "Nie mam informacji na ten temat w swojej bazie wiedzy."
Odpowiadaj po polsku, zwięźle i praktycznie."""

def build_prompt(query: str, results: list[dict]) -> str:
    if not results:
        return query  # brak kontekstu — LLM sam odpowie odmową

    context_parts = []
    for i, r in enumerate(results, 1):
        chunk = r["chunk"]
        context_parts.append(
            f"[{i}] {chunk['title']}\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    return f"""Kontekst z bazy wiedzy:
{context}

Pytanie użytkownika: {query}

Odpowiedz na podstawie powyższego kontekstu."""

# ── GENEROWANIE ──────────────────────────────────────────────────────────────

def generate_answer(prompt: str, client) -> str:
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.content[0].text

# ── PIPELINE ─────────────────────────────────────────────────────────────────

def ask(query, index, chunks, tokenizer, sess, input_names, client) -> dict:
    results = retrieve(query, index, chunks, tokenizer, sess, input_names)
    prompt  = build_prompt(query, results)
    answer = generate_answer(prompt, client)

    return {
        "query":   query,
        "answer":  answer,
        "sources": [r["chunk"]["title"] for r in results],
        "scores":  [r["score"] for r in results],
    }

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Inicjalizacja pipeline...\n")

    # Ładowanie komponentów
    index     = faiss.read_index(str(INDEX_FILE))
    chunks    = json.loads(META_FILE.read_text(encoding="utf-8"))
    tokenizer, sess, input_names = load_embedding_model(MODEL_DIR)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print(f"Indeks: {index.ntotal} wektorów | Model: {MODEL_DIR.name}\n")
    print("=" * 60)

    # Przykładowe pytania testowe
    test_queries = [
        "Jak często podlewać róże?",
        "Czym są objawy czarnej plamistości?",
        "Jak przygotować różę na zimę?",
        "Kiedy sadzić tulipany?",   # pytanie spoza bazy
    ]

    for query in test_queries:
        print(f"\n❓ {query}")
        result = ask(query, index, chunks, tokenizer, sess, input_names, client)

        print(f"\n💬 {result['answer']}")

        if result["sources"]:
            for title, score in zip(result["sources"], result["scores"]):
                print(f"   📄 {title} (score={score:.3f})")
        else:
            print("   ⚠️  Brak pasujących chunków powyżej progu — odpowiedź bez kontekstu")

        print("-" * 60)

if __name__ == "__main__":
    main()