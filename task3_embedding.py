import json

# Load the file
with open("knowledge_db_t3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Show available keys
print(data.keys())
facts = data.get("support", [])
print(f"Loaded {len(facts)} facts.")

# Create RAG-ready chunks (simple list of dicts)
chunks = [{"text": fact, "source": "knowledge_db_t3.json"} for fact in facts]


import numpy as np

# Embed the text chunks
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [chunk['text'] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# Save embeddings and chunks to file
np.save("fact_embeddings.npy", embeddings)

import json
with open("fact_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)



