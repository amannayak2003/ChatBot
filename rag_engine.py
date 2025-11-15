from utils.retriever import Retriever
from langchain_openai import ChatOpenAI

class RAGEngine:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = ChatOpenAI(model="gpt-4o-mini")   # you can switch to gpt-4o

    def generate_answer(self, query: str):
        # Step 1: Retrieve relevant chunks
        context_chunks = self.retriever.retrieve(query)
        context_text = "\n\n".join(context_chunks)

        # Step 2: Build prompt
        prompt = f"""
        You are an AI Research Assistant. Use the following context to answer the question.

        CONTEXT:
        {context_text}

        QUESTION:
        {query}

        Provide a clear and concise answer based ONLY on the context.
        """

        # Step 3: Generate answer
        response = self.llm.invoke(prompt)
        return response.content
