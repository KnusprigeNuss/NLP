import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.preprocessing import normalize
import gc
import torch

# truncates the text (not currently used since the token limit is not that tight)
def truncate_prompt(text, max_tokens=1000):
    input_ids = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(input_ids, skip_special_tokens=True)

# the classification function. Embedds the article the same way the facts are.
# then calculates the similarity and crafts a prompt.
def classify_with_rag(article_text, k=5):
    article_embedding = normalize(model_sbert.encode([article_text]), axis=1)
    D, I = index.search(article_embedding, k)
    similarities = [1 - (dist / 2) for dist in D[0]]
    retrieved = [
        f"[Similarity: {sim:.3f}] {chunks[idx]['text']}"
        for idx, sim in zip(I[0], similarities)
    ]

    context = "\n\n".join(retrieved)
    raw_prompt = f"""
        Answer with ONLY one word: "real" or "misinformation". 
        Based on the evidence, is the following article true or misinformation?
        Evidence:
        {context}

        Article:
        {article_text}
        """.strip()

    # prompt = truncate_prompt(raw_prompt)
    prompt = raw_prompt

    result = classifier(prompt, max_new_tokens=100)[0]['generated_text']
    print("LLM Output:", result)

    if "real" in result.lower() or "true" in result.lower():
        return 1
    elif "misinformation" in result.lower() or "false" in result.lower():
        return 0
    else:
        return -1 






with open("fact_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# embeddings of the facts
embeddings = np.load("fact_embeddings.npy")

dimension = embeddings.shape[1]  
embeddings = normalize(embeddings, axis=1)
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# model
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
classifier = pipeline("text2text-generation", model=bart_model, tokenizer=tokenizer)




with open("without_assessment.jsonl", 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)

# ground truth
df['Label'] = [1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,
               1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,1,
               1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0]

correct = 0
ambiguous = 0
reset = 0

for idx, row in df.iterrows():
    if reset >= 10:  
        gc.collect()
        torch.cuda.empty_cache()
        reset = 0

    reset += 1
    article_text = row['Text']
    label = classify_with_rag(article_text)

    if label == -1:
        ambiguous += 1
        print(f"Ambiguous classification at index {idx+1}")
        continue

    if label == df.at[idx, 'Label']:
        correct += 1
        print(f"Match at index {idx+1}")
    else:
        print(f"Mismatch at index {idx+1}")

# accuracy 
total_wo_ambiguous = len(df) - ambiguous
total = len(df)
print(f"\n Accuracy: {correct}/{total} = {correct / total * 100:.2f}%")
print(f"\n Accuracy wo ambigious: {correct}/{total_wo_ambiguous} = {correct / total_wo_ambiguous * 100:.2f}%")
if ambiguous > 0:
    print(f"Skipped {ambiguous} ambiguous cases.")
