from chunking import load_document, chunk_documents
from retriever import HybridRetriever
from ranker import Reranker
from rag import generate_simple_answer

def main():
    documents = load_document("data/documents.txt")
    chunks = chunk_documents(documents, max_sentences=2)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    retriever = HybridRetriever(chunks)
    reranker = Reranker()

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

if __name__ == "__main__":
    main()