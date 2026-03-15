from chunking import load_document, chunk_documents
from retriever import HybridRetriever
from ranker import Reranker
from rag import generate_simple_answer
from evaluation import recall_at_k, mrr
from eval_queries import EVAL_QUERIES

def main():
    documents = load_document("data/documents.txt")
    chunks = chunk_documents(documents, max_sentences=2)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    retriever = HybridRetriever(chunks)
    reranker = Reranker()

    print("\nRunning evaluation...")
    run_evaluation(retriever=retriever, reranker=reranker)

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        candidates = retriever.hybrid_search(query, top_k_sparse=5, top_k_dense=5, final_k=8)
        final_results = reranker.rerank(query, candidates, top_k = 5)

        print("\nTop results:")
        for i, r in enumerate(final_results, start=1):
            print(f"{i}. [doc {r['doc_id']}] score={r['rerank_score']:.4f}")
            print(f"    {r['text']}")

        answer = generate_simple_answer(query, final_results)
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)

def run_evaluation(retriever, reranker):
    recall_scores = []
    mrr_scores = []

    for item in EVAL_QUERIES:
        query = item["query"]
        relevant = item["relevant_doc_ids"]

        candidates = retriever.hybrid_search(query, top_k_sparse = 5, top_k_dense = 5, final_k = 8)
        final_results = reranker.rerank(query, candidates, top_k = 5)

        r = recall_at_k(final_results, relevant, k=5)
        m = mrr(final_results, relevant)

        recall_scores.append(r)
        mrr_scores.append(m)

        print(f"\nQuery: {query}")
        print(f"Recall@5: {r:.2f}")
        print(f"MRR: {m:.2f}")

    print("\n====================")
    print(f"Average Recall@5: {sum(recall_scores)/len(recall_scores):.3f}")
    print(f"Average MRR: {sum(mrr_scores)/len(mrr_scores):.3f}")
    print("====================")

if __name__ == "__main__":
    main()