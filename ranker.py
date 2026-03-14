from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)

        rescored = []
        for c, s in zip(candidates, scores):
            item = dict(c)
            item["rerank_score"] = float(s)
            rescored.append(item)

        rescored.sort(key=lambda x: x["rerank_score"], reverse = True)
        return rescored[:top_k]