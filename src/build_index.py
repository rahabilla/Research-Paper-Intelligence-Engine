import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "indexes/faiss.index"
EMBED_PATH = "indexes/embeddings.npy"

os.makedirs("indexes", exist_ok=True)

print("Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]

print("Building FAISS index...")
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
np.save(EMBED_PATH, embeddings)

print(f"Indexed {index.ntotal} chunks.")
