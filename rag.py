def build_context(top_chunks: list) -> str:
    return "\n".join([f"- {c['text']}" for c in top_chunks])

def generate_simple_answer(query: str, top_chunks: list) -> str:
    if not top_chunks:
        return "I could not find relevant information."
    
    context = build_context(top_chunks)

    answer = (
        f"Question: {query}\n\n"
        f"Relevant context: \n{context}\n\n"
        f"Provisional answer:\n"
        f"The answer is likely contained in the top retrieved passages above."
    )
    return answer