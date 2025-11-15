from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed(self, texts: list):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str):
        return self.embeddings.embed_query(query)
