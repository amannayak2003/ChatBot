from platform import system

import streamlit as st
import os
import time
from utils.loader import PDFLoader
from utils.cleaner import TextCleaner
from utils.chunker import TextChunker
from utils.embedding import EmbeddingGenerator
from utils.vectorstore import VectorStore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_rag_answer(query, vectorstore):
    # 1. Retrieve top chunks
    results = vectorstore.search(query, top_k=3)
    context = "\n\n".join(results) if results else "No relevant context found."

    # 2. Build RAG prompt
    prompt = f"""
    You are an AI Research Assistant.
    Use ONLY the context below to answer the question.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    Answer clearly, concisely, and factually.
    """

    # 3. LLM call
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(prompt)

    return response.content


st.set_page_config(page_title="AI Assistant", layout="wide")

if "vectorstore" not in st.session_state:
    print("Initializing VectorStore...")
    st.session_state["vectorstore"] = VectorStore()

with st.sidebar:
    st.title("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or Text files",
        type=["pdf","docx","txt"],
        accept_multiple_files=True
    )

    if st.button("Continue"):
        if uploaded_files:
            st.session_state["process_docs"] = True
        else:
            st.toast("Please upload at least one document.", icon="⚠️")

st.title("📚 AI Assistant")

tab_chat, tab_summary, tab_citations = st.tabs(["💬 Chat", "📝 Summary", "📌 Citations"])

if st.session_state.get("process_docs", False):
    status_placeholder = st.empty()
    # START STATUS
    with status_placeholder.status("Processing your documents...", expanded=False) as status:

        # 1. Create folder
        os.makedirs("uploaded_pdfs", exist_ok=True)

        # 2. Save PDFs
        pdf_paths = []
        for f in uploaded_files:
            path = f"uploaded_pdfs/{f.name}"
            with open(path, "wb") as pdf:
                pdf.write(f.read())
            pdf_paths.append(path)

        # 3. Load
        loader = PDFLoader(pdf_paths)
        docs = loader.load_pdfs()

        # 4. Clean & chunk
        cleaner = TextCleaner()
        chunker = TextChunker()

        chunks = []
        for d in docs:
            clean = cleaner.clean(d["text"])
            chunks.extend(chunker.chunk_text(clean))

        # Prevent crash
        if len(chunks) == 0:
            status.update(label="❌ No text found. Cannot process.", state="error")
            st.session_state["process_docs"] = False
            st.stop()

        # 5. Embed
        embedder = EmbeddingGenerator()
        embeddings = embedder.embed(chunks)

        # 6. Store
        store = st.session_state["vectorstore"]
        store.add(chunks)

        # SUCCESS MESSAGE (THIS WILL SHOW!)
        status.update(
            label="Documents processed successfully!",
            state="complete"
        )

        time.sleep(1.5)

        status_placeholder.empty()

    # Set values AFTER status finishes (outside status block)
    st.session_state["chunks"] = chunks
    st.session_state["process_docs"] = False

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a creative branding expert. Your job is to craft short, catchy, and modern subheadings for digital products. "
     "Make it sound premium, memorable, and friendly. Output only the subheading, without quotes."),

    ("human",
     "Create a cool subheading for my AI chatbot that helps users with answers, assistance, and conversations.")
])


formatted_messages = prompt.format_messages()


response = llm.invoke(formatted_messages)

subHeader = response.content.strip('"')



with tab_chat:
    st.subheader(subHeader)

    question = st.text_input("Ask anything")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                store = st.session_state["vectorstore"]
                answer = generate_rag_answer(question, store)

            st.write("Answer:")
            st.write(answer)

# --------------------------------------------------------------------------------
# SUMMARY TAB
# --------------------------------------------------------------------------------
with tab_summary:
    st.subheader("Document Summary")
    st.info("Summary will appear here after we integrate the summarizer agent.")

# --------------------------------------------------------------------------------
# CITATIONS TAB
# --------------------------------------------------------------------------------
with tab_citations:
    st.subheader("Citations")
    st.info("Citations will be extracted after we integrate the citation agent.")
