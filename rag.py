from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from dotenv import load_dotenv
import datetime
from langchain.agents import tool

load_dotenv()

# Get current time
@tool
def get_current_time(input: str) -> str:
    """Returns the current time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")



# LLM
model = ChatOpenAI(model="gpt-4o")

# Pull prompt from hub
prompt = hub.pull("hwchase17/react")

query = "what is the current time in london by the way you are in India so calculate accroding to that?"
tools = [get_current_time]     # no tools

# Correct arguments for old API
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke({"input": query})

# print(response["output"])
