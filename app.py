import streamlit as st
import os
import uuid

from utils.loader import PDFLoader
from utils.cleaner import TextCleaner
from utils.chunker import TextChunker
from utils.embedding import EmbeddingGenerator
from utils.vectorstore import VectorStore

from langchain_openai import ChatOpenAI

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
# --------------------------------------------------------------------------------

st.set_page_config(page_title="AI Research Assistant", layout="wide")

# --------------------------------------------------------------------------------
# Initialize VectorStore once (important)
# --------------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    print("Initializing VectorStore...")
    st.session_state["vectorstore"] = VectorStore()

# --------------------------------------------------------------------------------
# SIDEBAR UI
# --------------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload Research PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            st.session_state["process_docs"] = True
        else:
            st.warning("Please upload at least one PDF.")

# --------------------------------------------------------------------------------
# MAIN TITLE
# --------------------------------------------------------------------------------
st.title("📚 AI Research Assistant")

tab_chat, tab_summary, tab_citations = st.tabs(["💬 Chat", "📝 Summary", "📌 Citations"])

# --------------------------------------------------------------------------------
# PROCESSING PIPELINE
# --------------------------------------------------------------------------------
if st.session_state.get("process_docs"):

    st.info("Processing documents... Please wait.")

    # 0. Create unique directory for uploaded PDFs
    if not os.path.exists("uploaded_pdfs"):
        os.makedirs("uploaded_pdfs")   

    # 1. Save uploaded PDFs
    pdf_paths = []
    for f in uploaded_files:
        path = f"uploaded_pdfs/{f.name}"
        with open(path, "wb") as pdf:
            pdf.write(f.read())
        pdf_paths.append(path)

    print("PDFs saved locally:", pdf_paths)

    # 2. Load PDFs
    print("Loading PDFs...")
    loader = PDFLoader(pdf_paths)
    docs = loader.load_pdfs()
    print("Loaded documents count:", len(docs))

    # 3. Clean + Chunk
    print("Cleaning and chunking...")
    cleaner = TextCleaner()
    chunker = TextChunker()

    chunks = []
    for d in docs:
        clean_text = cleaner.clean(d["text"])
        file_chunks = chunker.chunk_text(clean_text)
        chunks.extend(file_chunks)

    print("Total chunks generated:", len(chunks))

    # 4. Embed & Store
    print("Generating embeddings...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed(chunks)
    print("Embedding generation complete.")

    # 4. Store in FAISS Vector DB
    print("Adding embeddings to vector DB (FAISS)...")
    store = st.session_state["vectorstore"]
    store.add(chunks)
    print("Embeddings added successfully!")



    print("Database storage complete.")

    # Final success
    st.success("🎉 Documents processed successfully!")
    st.session_state["chunks"] = chunks

# -----------------------------------
# CHAT TAB
# -----------------------------------
with tab_chat:
    st.subheader("Ask a Question")

    question = st.text_input("Enter your query:")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                store = st.session_state["vectorstore"]
                answer = generate_rag_answer(question, store)

            st.write("### 📘 Answer:")
            st.write(answer)

            st.write("### 🔍 Context Used:")
            for chunk in store.search(question, top_k=3):
                st.info(chunk)
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
