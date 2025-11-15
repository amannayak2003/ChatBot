from utils.embedding import EmbeddingGenerator
from dotenv import load_dotenv

load_dotenv()
embedder = EmbeddingGenerator()

print("Trying to embed...")

result = embedder.embed(["Hello world, this is a test chunk."])

print("Embedding result:", result)
