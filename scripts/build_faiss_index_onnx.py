import json
import numpy as np
from pathlib import Path
import faiss
import onnxruntime as ort
from transformers import AutoTokenizer
import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

CHUNKS_FILE = Path("knowledge/chunks/poradnik_pielegnacji_roz_chunks_fixed.jsonl")
INDEX_FILE  = Path("knowledge/faiss/rose_index.faiss")
META_FILE   = Path("knowledge/faiss/rose_chunks_meta.json")
MODEL_DIR   = Path("models/onnx/e5-small-int8")

def load_chunks(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clip(min=1e-9)
    return pooled

def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms.clip(min=1e-9)

def encode(sess, tokenizer, texts, input_names, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )
        feed = {k: v for k, v in encoded.items() if k in input_names}
        if "token_type_ids" in input_names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(encoded["input_ids"])

        outputs = sess.run(None, feed)
        pooled  = mean_pool(outputs[0], encoded["attention_mask"])
        all_embeddings.append(pooled)
        print(f"  batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    return normalize(np.vstack(all_embeddings))

def main():
    print("Ładowanie chunków...")
    chunks = load_chunks(CHUNKS_FILE)

    # ✅ Prefix passage:
    texts = [
        f"passage: {c.get('title', '')}. {c.get('text', '')} "
        f"Słowa kluczowe: {', '.join(c.get('keywords', []))}"
        for c in chunks
    ]
    texts = [t for t in texts if len(t.strip()) > 20]
    print(f"Chunków: {len(texts)}")

    print("Ładowanie modelu ONNX...")
    tokenizer   = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    sess        = ort.InferenceSession(
        str(MODEL_DIR / "model.onnx"),
        providers=["CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in sess.get_inputs()]

    print("Tworzenie embeddingów...")
    embeddings = encode(sess, tokenizer, texts, input_names)

    print(f"Wymiar: {embeddings.shape[1]}, chunków: {embeddings.shape[0]}")

    print("Budowa indeksu FAISS...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))

    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Indeks gotowy — {index.ntotal} wektorów")
    print(f"   {INDEX_FILE}")
    print(f"   {META_FILE}")

if __name__ == "__main__":
    main()