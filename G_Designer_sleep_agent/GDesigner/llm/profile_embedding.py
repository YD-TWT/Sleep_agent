from sentence_transformers import SentenceTransformer

_embedding_model = None

def get_sentence_embedding(sentence):
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    embeddings = _embedding_model.encode(sentence)
    return embeddings
