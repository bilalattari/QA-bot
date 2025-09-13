# src/retriever.py
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd

class Retriever:
    def __init__(self, dataset_path, faiss_index_path, meta_path, model_name, device="cpu"):
        self.df = pd.read_csv(dataset_path)
        self.index = faiss.read_index(str(faiss_index_path))
        self.meta = json.load(open(meta_path, encoding="utf-8"))
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def search(self, query, top_k=5):
        query_vec = self.embed([query]).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.meta[idx]
            answer_row = self.df[self.df["id"] == meta["id"]].iloc[0]
            results.append({
                "id": int(answer_row["id"]),
                "question": str(answer_row["question"]),
                "answer": str(answer_row["answer"]),
                "score": float(score)
            })
        return results
