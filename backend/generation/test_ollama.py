import requests

prompt = "Who is Darth Vader?"

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
)

result = response.json()

print("\nLlama 3 Response:\n")

print(result["response"])