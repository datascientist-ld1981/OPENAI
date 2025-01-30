import streamlit as st
import pdfplumber
import os
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle cases where no text is found
    return text

# Function to split text into chunks
def chunk_text(text, chunk_size=500, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.create_documents([text])

# Function to create FAISS vector store using free Hugging Face embeddings
def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Free model
    texts = [chunk.page_content for chunk in text_chunks]
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    return vector_store

# Function to retrieve answers (Mock response for testing)
def get_rag_response(query, vector_store):
    # Mock responses for common questions
    sample_responses = {
        "What is the main idea of the paper?": "The paper discusses advancements in AI-driven sentiment analysis.",
        "What methods are used in the research?": "It uses Naive Bayes, RNNs, and Transformer models for comparison.",
        "What are the key findings?": "The study found that BERT outperforms traditional models in contextual understanding.",
    }
    return sample_responses.get(query, "No relevant information found in the document.")

# Streamlit UI
st.title("📄 AI Research Paper Q&A (No OpenAI API)")
st.sidebar.header("Upload Research Paper")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        if not pdf_text.strip():
            st.error("No text found in the PDF. Try another file.")
        else:
            text_chunks = chunk_text(pdf_text)
            vector_store = create_vector_store(text_chunks)
            st.sidebar.success("PDF processed successfully!")

            query = st.text_input("Ask a question about the paper:")
            if st.button("Get Answer") and query:
                with st.spinner("Generating answer..."):
                    answer = get_rag_response(query, vector_store)
                    st.write("### Answer:")
                    st.write(answer)
