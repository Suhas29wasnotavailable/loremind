from pathlib import Path

FILE_PATH = "data/starwars_lore.txt"

with open(FILE_PATH, "r", encoding="utf-8") as file:
    text = file.read()

chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

print("\nGenerated Chunks:\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(chunk)
    print("-" * 50)