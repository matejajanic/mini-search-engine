import numpy as np
import faiss

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, chunks: List[dict], embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]

        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # sparse retriever
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)

        # dense retriever
        self.embedder = SentenceTransformer(embedding_model_name)
        self.embeddings = self.embedder.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 50
        self.index.add(self.embeddings)

    def sparse_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)

        ids = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ids]
    
    def dense_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        scores, ids = self.index.search(q_emb, top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
    
    def hybrid_search(self, query: str, top_k_sparse: int = 5, top_k_dense: int = 5, final_k: int = 5):
        sparse_results = self.sparse_search(query, top_k_sparse)
        dense_results = self.dense_search(query, top_k_dense)

        combined = {}

        for rank, (idx, score) in enumerate(sparse_results, start=1):
            combined.setdefault(idx, 0.0)
            combined[idx] += 1.0 / (60 + rank)

        for rank, (idx, score) in enumerate(dense_results, start=1):
            combined.setdefault(idx, 0.0)
            combined[idx] += 1.0 / (60 + rank)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse = True)[:final_k]

        return [
            {
                "chunk_id": idx,
                "score": score,
                "text": self.chunks[idx]["text"],
                "doc_id": self.chunks[idx]["doc_id"]
            }
            for idx, score in ranked
        ]