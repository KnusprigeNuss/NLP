from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

import json
import os

import nltk
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

nltk.download("punkt")

INDEX_PATH = "rag_index"
MODEL_PATH = r"C:\Users\Manuel\Downloads\mistral-7b-v0.1.Q4_K_M.gguf"

# Load or create FAISS vector store
def load_or_create_vectorstore():
    if os.path.exists(INDEX_PATH):
        print("Loading existing vector store...")
        return FAISS.load_local(INDEX_PATH, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

    print("Creating vector store from rag/ documents...")
    docs = []
    for filename in os.listdir("rag/"):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join("rag", filename), encoding="utf-8")
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

# Load LLaMA.cpp-based LLM
def load_local_llm():
    print("Loading Mistral model via llama-cpp-python...")
    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0,        # deterministic output, good for classification
        max_tokens=5000,        # increased max tokens for more reliable output
        top_p=0.9,
        n_ctx=2048,
        n_threads=4,
        f16_kv=True,
        verbose=False,
        stop=["Answer: SUPPORTED", "Answer: REFUTED", "Answer: UNCERTAIN"]
    )

# Setup RAG pipeline
def setup_rag():
    vectorstore = load_or_create_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # increased k to 5
    llm = load_local_llm()

    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return rag_chain

# Extract climate-related factual claims using zero-shot classifier
def extract_climate_claims(text):
    sentences = nltk.sent_tokenize(text)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["climate-related factual claim", "opinion", "not a claim"]

    claims = []
    for sentence in sentences:
        result = classifier(sentence, candidate_labels=candidate_labels)
        if result["labels"][0] == "climate-related factual claim":
            claims.append(sentence)
    return claims

# Classify a single claim as SUPPORTED, REFUTED, or UNCERTAIN using RAG+LLM
def classify_claim(rag_chain, claim_text):
    prompt = f"""
You are a climate science expert.
Classify the following claim as SUPPORTED, REFUTED, or UNCERTAIN based on current scientific consensus.
Respond with only one word.
Claim: {claim_text}
Answer:
"""
    output = rag_chain.run(prompt).strip().upper()
    label = output.split()[0] if output.split() else "UNCERTAIN"
    if label not in ["SUPPORTED", "REFUTED", "UNCERTAIN"]:
        label = "UNCERTAIN"

    print("\n----- Prompt Sent to LLM -----")
    print(prompt)
    print("----- Raw Model Output -----")
    print(output)
    print(f"Prediction: {label}")
    return label

# Main execution
if __name__ == "__main__":
    rag_chain = setup_rag()

    # Load article data
    with open("data/without_assessment.jsonl", 'r') as file:
        data = [json.loads(line) for line in file]
    article = data[0]['Text']

    print("\nExtracting climate-related factual claims...\n")
    claims = extract_climate_claims(article)
    print(f"Found {len(claims)} claims.\n")

    for i, claim in enumerate(claims):
        print(f"\n--- Claim {i + 1} ---")
        print(claim)
        classify_claim(rag_chain, claim)
