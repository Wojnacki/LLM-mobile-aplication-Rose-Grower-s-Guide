import faiss
import numpy as np

# Load the FAISS index
index = faiss.read_index('assets/rose_index.faiss')

print(f"Index type: {type(index)}")
print(f"n: {index.ntotal}")
print(f"dim: {index.d}")
print(f"Is trained: {index.is_trained}")

# If it's IndexFlat, print some vectors
if isinstance(index, faiss.IndexFlat):
    vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)
    print(f"First vector: {vectors[0][:10]}")  # first 10 dims
    print(f"Last vector: {vectors[-1][:10]}")  # last 10 dims