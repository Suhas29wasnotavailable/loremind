from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "starwars_lore"

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# User query
query = input("\nEnter your lore query: ")

# Convert query into embedding
query_embedding = model.encode(query).tolist()

# Search Qdrant
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=3
).points

# Display results
print("\nTop Semantic Matches:\n")

for i, result in enumerate(results):
    print(f"Result {i+1}")
    print(f"Score: {result.score:.4f}")
    print(result.payload["text"])
    print("-" * 50)