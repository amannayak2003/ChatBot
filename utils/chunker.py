from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def chunk_text(self, text: str) -> list:
        return self.text_splitter.split_text(text)    