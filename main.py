from langchain_openai import ChatOpenAI as chatOpenai
from dotenv import load_dotenv

load_dotenv()

llm = chatOpenai(model="gpt-4o")
response = llm.invoke("Question")

print(response.content)