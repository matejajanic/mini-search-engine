from typing import List

def load_document(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    docs = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
    return docs

def chunk_documents(documents: List[str], max_sentences: int = 2):
    chunks = []
    chunk_id = 0

    for doc_id, doc in enumerate(documents):
        sentences = [s.strip() for s in doc.split(".") if s.strip()]
        for i in range(0, len(sentences), max_sentences):
            piece = ". ".join(sentences[i:i+max_sentences]).strip()
            if piece:
                if not piece.endswith("."):
                    piece += "."
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": piece
                })

                chunk_id += 1
        
    return chunks