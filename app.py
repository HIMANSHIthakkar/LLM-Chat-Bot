import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Set your Groq API Key here 
os.environ["GROQ_API_KEY"] = "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"  # <-- Replace with your actual key


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to create FAISS vector store
def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)


# Load FAISS vector store
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


# Build QA Chain using Groq's supported model (llama3-8b-8192)
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()

    #  Use Groq's latest supported model
    llm = ChatOpenAI(
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        model_name="llama3-8b-8192",  # ✅ valid model name
        temperature=0.3
    )

    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    return qa_chain


# Streamlit App
st.set_page_config(page_title="PDF Q&A Chatbot", layout="centered")
st.title(" PDF Chatbot with FAISS + Groq (LLaMA3)")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("📄 Upload your PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = f"uploaded/{uploaded_file.name}"
    os.makedirs("uploaded", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(pdf_path)

    st.info(" Creating FAISS vector store...")
    create_faiss_vector_store(text)

    st.info(" Initializing Groq LLM chatbot...")
    qa_chain = build_qa_chain()
    st.success("✅ Chatbot is ready! Ask your question below.")

# Question Input
if 'qa_chain' in locals():
    question = st.text_input("💬 Ask a question about your PDF:")
    if question:
        st.info(" Generating answer...")
        try:
            answer = qa_chain.run(question)
            st.success(f" Answer: {answer}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
