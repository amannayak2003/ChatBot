from langchain_community.vectorstores import FAISS
from utils.embedding import EmbeddingGenerator

class VectorStore:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.store = None   # will hold FAISS index

    def add(self, texts):
        embeddings = self.embedder.embed(texts)

        # Create FAISS index first time
        if self.store is None:
            self.store = FAISS.from_texts(
                texts,
                embedding=self.embedder.embeddings
            )
        else:
            # Add new texts + embeddings
            self.store.add_texts(
                texts,
                embeddings=embeddings
            )

    def search(self, query, top_k=5):
        if self.store is None:
            return []

        query_embedding = self.embedder.embed_query(query)
        docs = self.store.similarity_search_by_vector(query_embedding, k=top_k)

        return [doc.page_content for doc in docs]
