import os
import faiss
import pickle
import openai
from tqdm import tqdm
from typing import List
import tiktoken
import streamlit as st
from openai import OpenAI
import numpy as np

openai.api_key = st.secrets["OPENAI_API_KEY"]


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


# Load your regulations from .txt files
def load_regulations(folder_path: str):
    docs = []
    metadata = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            location = filename.replace('.txt', '')
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = chunk_text(text)
                for chunk in chunks:
                    docs.append(chunk)
                    metadata.append({'location': location})
    return docs, metadata

# Chunking function
def chunk_text(text, max_tokens=300):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        decoded_chunk = tokenizer.decode(chunk)
        chunks.append(decoded_chunk)
    return chunks

# Embedding and indexing
def build_index(docs: List[str]):
    embeddings = [get_embedding(doc, model="text-embedding-3-small") for doc in tqdm(docs)]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

# Save everything
def save_index(index, docs, metadata):
    faiss.write_index(index, "regulations.index")
    with open("regulations_meta.pkl", "wb") as f:
        pickle.dump((docs, metadata), f)

# === RUN THIS ===
docs, meta = load_regulations("regulations/")
index, embeddings = build_index(docs)
save_index(index, docs, meta)