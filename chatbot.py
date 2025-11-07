import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# --- Streamlit UI ---
st.title("LangChain Chatbot with Streamlit")m

# Initialize chat model
llm = ChatOpenAI(model="gpt-4o")

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant.")
    ]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat input box
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    human_message = HumanMessage(content=user_input)
    st.session_state.messages.append(human_message)
    st.chat_message("user").write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(st.session_state.messages)
            st.write(response.content)

    # Save AI response
    ai_message = AIMessage(content=response.content)
    st.session_state.messages.append(ai_message)
