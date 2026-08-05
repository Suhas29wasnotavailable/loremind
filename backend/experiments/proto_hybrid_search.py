from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

COLLECTION_NAME = "starwars_lore"

with open("data/starwars_lore.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# -----------------------------
# BM25 SETUP
# -----------------------------

tokenized_chunks = [chunk.lower().split() for chunk in chunks]

bm25 = BM25Okapi(tokenized_chunks)

# -----------------------------
# VECTOR SEARCH SETUP
# -----------------------------

client = QdrantClient(host="localhost", port=6333)

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -----------------------------
# USER QUERY
# -----------------------------

query = input("\nEnter your hybrid query: ")

# -----------------------------
# VECTOR SEARCH
# -----------------------------

query_embedding = model.encode(query).tolist()

vector_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=5
).points

# Convert vector results into dict
vector_scores = {}

for result in vector_results:
    chunk_text = result.payload["text"]
    vector_scores[chunk_text] = result.score

# -----------------------------
# BM25 SEARCH
# -----------------------------

tokenized_query = query.lower().split()

bm25_scores_raw = bm25.get_scores(tokenized_query)

# Normalize BM25 scores : because BM25 normally dominates.
max_bm25 = max(bm25_scores_raw)

bm25_scores = {}

for chunk, score in zip(chunks, bm25_scores_raw):

    normalized_score = score / max_bm25 if max_bm25 > 0 else 0

    bm25_scores[chunk] = normalized_score
# -----------------------------
# HYBRID SCORE COMBINATION
# -----------------------------

hybrid_scores = {}

all_chunks = set(vector_scores.keys()) | set(bm25_scores.keys())

for chunk in all_chunks:

    vector_score = vector_scores.get(chunk, 0)
    bm25_score = bm25_scores.get(chunk, 0)

    # Weighted combination
    hybrid_score = (0.7 * vector_score) + (0.3 * bm25_score)

    hybrid_scores[chunk] = hybrid_score

# -----------------------------
# SORT RESULTS
# -----------------------------

ranked_results = sorted(
    hybrid_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("\nTop Hybrid Retrieval Results:\n")

for i, (chunk, score) in enumerate(ranked_results[:5]):

    print(f"Result {i+1}")
    print(f"Hybrid Score: {score:.4f}")
    print(chunk)
    print("-" * 50)