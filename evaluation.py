def recall_at_k(results, relevant_doc_ids, k=5):
    retrieved = {r["doc_id"] for r in results[:k]}
    relevant = set(relevant_doc_ids)

    if not relevant:
        return 0
    
    return len(retrieved & relevant) / len(relevant)

def mrr(results, relevant_doc_ids):
    relevant = set(relevant_doc_ids)

    for i, r in enumerate(results, start=1):
        if r["doc_id"] in relevant:
            return 1/i
        
    return 0