import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "indexes/faiss.index"

TOP_K = 10

print("Loading data...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

index = faiss.read_index(INDEX_PATH)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------
# DEFINE TEST QUERIES WITH GROUND TRUTH KEYWORDS
# ---------------------------------------------
test_queries = [
    {
        "query": "future work",
        "expected_keywords": ["future", "challenge", "direction"]
    },
    {
        "query": "evaluation metrics",
        "expected_keywords": ["evaluation", "metric", "benchmark"]
    },
    {
        "query": "retriever architecture",
        "expected_keywords": ["retriever", "indexing", "embedding"]
    }
]


def compute_recall_at_k(retrieved_texts, keywords):
    for text in retrieved_texts:
        if any(k in text.lower() for k in keywords):
            return 1
    return 0


recalls = []
mrrs = []

for test in test_queries:
    query = test["query"]
    keywords = test["expected_keywords"]

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, TOP_K)

    retrieved = [chunks[i]["text"] for i in indices[0]]

    # Recall@K
    recall = compute_recall_at_k(retrieved, keywords)
    recalls.append(recall)

    # MRR
    rr = 0
    for rank, text in enumerate(retrieved):
        if any(k in text.lower() for k in keywords):
            rr = 1 / (rank + 1)
            break

    mrrs.append(rr)

print("\nEvaluation Results:")
print("Recall@10:", sum(recalls) / len(recalls))
print("MRR:", sum(mrrs) / len(mrrs))
