from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading the model...")


model = SentenceTransformer("BAAI/bge-small-en-v1.5") 

text1 = "Anakin Skywalker became Darth Vader."
text2 = "Vader was once a Jedi named Anakin."
text3 = "The Death Star was a massive superweapon."

print("\nGenerating the word embeddings...")

emb1 = model.encode(text1) 
emb2 = model.encode(text2)
emb3 = model.encode(text3)

print("\nEmbedding shape: ")
print(emb1.shape) 

print("\nSemantic similarities:")
similarity_1_2 = cosine_similarity([emb1],[emb2])[0][0]
similarity_1_3 = cosine_similarity([emb1],[emb3])[0][0]

print(f"Text1 vs Text2: {similarity_1_2:.4f}")
print(f"Text1 vs Text3: {similarity_1_3:.4f}")