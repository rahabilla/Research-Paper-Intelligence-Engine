import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "indexes/faiss.index"

TOP_K_PER_QUERY = 15
FINAL_TOP_K = 10

print("Loading data...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

index = faiss.read_index(INDEX_PATH)

print("Loading models...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# -------------------------------------------------
# MULTI QUERY EXPANSION
# -------------------------------------------------
def expand_query(query):
    variations = [
        query,
        query.replace("future directions", "future work"),
        query.replace("future directions", "future challenges"),
        query.replace("future directions", "research outlook"),
        query.replace("future directions", "open problems")
    ]
    return list(set(variations))


while True:
    query = input("\nEnter query (or 'exit'): ")
    if query.lower() == "exit":
        break

    expanded_queries = expand_query(query)

    all_candidate_indices = set()

    # ---------------------------------------------
    # DENSE RETRIEVAL FOR EACH QUERY VARIATION
    # ---------------------------------------------
    for q in expanded_queries:
        q_embedding = embed_model.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = index.search(q_embedding, TOP_K_PER_QUERY)

        for idx in indices[0]:
            all_candidate_indices.add(idx)

    all_candidate_indices = list(all_candidate_indices)

    candidate_chunks = [chunks[i]["text"] for i in all_candidate_indices]

    # ---------------------------------------------
    # CROSS-ENCODER RERANK
    # ---------------------------------------------
    pairs = [(query, chunk) for chunk in candidate_chunks]
    ce_scores = reranker.predict(pairs)

    total_chunks = len(chunks)

    reranked = []
    for idx, ce_score in zip(all_candidate_indices, ce_scores):
        # position boost (later chunks slightly boosted)
        position_boost = idx / total_chunks
        final_score = ce_score + 0.05 * position_boost
        reranked.append((chunks[idx]["text"], final_score))

    reranked.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Results:\n")
    for rank, (text, score) in enumerate(reranked[:FINAL_TOP_K]):
        print(f"Rank {rank+1} | Score: {score:.4f}")
        print(text[:400])
        print("-" * 80)
