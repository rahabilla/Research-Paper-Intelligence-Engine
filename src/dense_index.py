import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config
# ---------------------------
CHUNKS_PATH = "data/chunks.json"
FAISS_INDEX_PATH = "indexes/faiss.index"
EMBEDDINGS_PATH = "indexes/embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs("indexes", exist_ok=True)

# ---------------------------
# Load Model
# ---------------------------
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ---------------------------
# Load Chunks
# ---------------------------
print("Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

# ---------------------------
# Generate Embeddings
# ---------------------------
print("Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # IMPORTANT for cosine similarity
)

dimension = embeddings.shape[1]

# ---------------------------
# Create FAISS Index
# ---------------------------
print("Creating FAISS index...")
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine (since normalized)
index.add(embeddings)

# ---------------------------
# Save
# ---------------------------
faiss.write_index(index, FAISS_INDEX_PATH)
np.save(EMBEDDINGS_PATH, embeddings)

print(f"\nFAISS index saved to {FAISS_INDEX_PATH}")
print(f"Embeddings saved to {EMBEDDINGS_PATH}")
print(f"Total vectors indexed: {index.ntotal}")
