from utils.vectorstore import VectorStore

class Retriever:
    def __init__(self, top_k=3):
        self.store = VectorStore()
        self.top_k = top_k

    def retrieve(self, query: str):
        results = self.store.search(query, top_k=self.top_k)
        return [doc.page_content for doc in results]
