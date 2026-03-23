import json
import numpy as np
import onnxruntime as ort
import faiss

# Load tokenizer
with open('assets/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)

vocab = {}
for item in tokenizer_data['model']['vocab']:
    vocab[item[0]] = item[1]

# Normalization function (same as in Dart)
def normalize_polish(text):
    table = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ż': 'z', 'ź': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N', 'Ó': 'O', 'Ś': 'S', 'Ż': 'Z', 'Ź': 'Z',
    }
    return ''.join(table.get(c, c) for c in text)

# Tokenization function (simplified version of Dart logic)
def tokenize(text):
    cls_id = 0  # <s>
    sep_id = 2  # </s>
    unk_id = 3  # <unk>
    max_len = 512

    words = text.lower().split()
    ids = [cls_id]

    def lookup_token(token):
        if token in vocab:
            return vocab[token]
        normalized = normalize_polish(token)
        if normalized != token and normalized in vocab:
            return vocab[normalized]
        return None

    for word in words:
        if not word:
            continue

        norm_word = normalize_polish(word)
        candidates = [f'▁{word}', f'▁{norm_word}', word, norm_word]

        added = False
        for cand in candidates:
            token_id = lookup_token(cand)
            if token_id is not None:
                ids.append(token_id)
                added = True
                break

        if added:
            if len(ids) >= max_len - 1:
                break
            continue

        # Subword tokenization (simplified)
        found = False
        for length in range(len(word), 0, -1):
            sub = f'▁{word[:length]}' if length == len(word) else word[:length]
            sub_norm = f'▁{norm_word[:length]}' if length == len(word) else norm_word[:length]
            sub_id = lookup_token(sub) or lookup_token(sub_norm)
            if sub_id is not None:
                ids.append(sub_id)
                remaining = word[length:]
                while remaining:
                    matched = False
                    for l in range(len(remaining), 0, -1):
                        s = remaining[:l]
                        ns = normalize_polish(remaining[:l])
                        rest_id = lookup_token(s) or lookup_token(ns)
                        if rest_id is not None:
                            ids.append(rest_id)
                            remaining = remaining[l:]
                            matched = True
                            break
                    if not matched:
                        ids.append(unk_id)
                        break
                found = True
                break

        if not found:
            ids.append(unk_id)

        if len(ids) >= max_len - 1:
            break

    ids.append(sep_id)
    return ids

# Load ONNX model
session = ort.InferenceSession('assets/model.onnx')

# Load chunks
with open('assets/rose_chunks_meta.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

embeddings = []

for chunk in chunks:
    title = chunk.get('title', '')
    text = chunk.get('text', '')
    keywords = ', '.join(chunk.get('keywords', []))
    full_text = f'{title}\n{text}\nKeywords: {keywords}'

    # Tokenize
    input_ids = tokenize(f'query: {full_text}')
    seq_len = len(input_ids)

    # Prepare inputs
    input_ids_tensor = np.array([input_ids], dtype=np.int64)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    token_type_ids = np.zeros((1, seq_len), dtype=np.int64)

    # Run model
    outputs = session.run(None, {
        'input_ids': input_ids_tensor,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
    })

    # Mean pooling
    last_hidden_state = outputs[0]
    token_embeddings = last_hidden_state[0]
    emb_dim = token_embeddings.shape[1]
    pooled = np.mean(token_embeddings, axis=0)

    # L2 normalize
    norm = np.linalg.norm(pooled)
    normalized_emb = pooled / norm
    embeddings.append(normalized_emb)

# Create FAISS index
embeddings_array = np.array(embeddings, dtype=np.float32)
dim = embeddings_array.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_array)

# Save index
faiss.write_index(index, 'assets/rose_index.faiss')
print(f'Saved FAISS index with {len(embeddings)} vectors of dimension {dim}')