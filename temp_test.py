import json
import numpy as np
from pathlib import Path
import faiss
import onnxruntime as ort
from transformers import AutoTokenizer
import logging

logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)

INDEX_FILE = Path('mobile/flutter_app/assets/rose_index.faiss')
META_FILE = Path('mobile/flutter_app/assets/rose_chunks_meta.json')
MODEL_DIR = Path('models/onnx/e5-small-int8')

def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    sess = ort.InferenceSession(str(model_dir / 'model.onnx'), providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in sess.get_inputs()]
    return tokenizer, sess, input_names

def encode_query(query: str, tokenizer, sess, input_names) -> np.ndarray:
    encoded = tokenizer(f'query: {query}', return_tensors='np', padding=True, truncation=True, max_length=512)
    feed = {k: v for k, v in encoded.items() if k in input_names}
    if 'token_type_ids' in input_names and 'token_type_ids' not in feed:
        feed['token_type_ids'] = np.zeros_like(encoded['input_ids'])
    outputs = sess.run(None, feed)
    token_embeddings = outputs[0]
    mask = encoded['attention_mask'][..., np.newaxis].astype(np.float32)
    pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clip(min=1e-9)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / norm.clip(min=1e-9)).astype(np.float32)

index = faiss.read_index(str(INDEX_FILE))
chunks = json.loads(META_FILE.read_text(encoding='utf-8'))
tokenizer, sess, input_names = load_model(MODEL_DIR)

query = 'jakie są rodzaje róż'
q_emb = encode_query(query, tokenizer, sess, input_names)
scores, indices = index.search(q_emb, 5)

print(f'Query: {query}')
for rank, idx in enumerate(indices[0], 1):
    chunk = chunks[idx]
    score = scores[0][rank - 1]
    print(f'[{rank}] score={score:.3f} | {chunk["title"]}')
    print(f'   Text: {chunk["text"][:200]}...')
    print()