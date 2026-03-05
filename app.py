import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("PDF RAG Chatbot (Groq + LLaMA-3)")
st.write("Ask questions from your PDF using Retrieval Augmented Generation.")


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()


uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        st.error("No text extracted from PDF")
        st.stop()

    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(text)
    st.success(f"PDF split into {len(chunks)} chunks")

    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)

    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    
    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Thinking..."):
            docs = vectorstore.similarity_search(query, k=3)

            if not docs:
                st.warning("I don't know")
            else:
                context = "\n\n".join(doc.page_content for doc in docs)

                prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

                response = llm.invoke(prompt)

                st.subheader("Answer")
                st.write(response.content)
