import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page setup
st.set_page_config(page_title="Free RAG Assistant", page_icon="📚")
st.title("📚 CampusGuide / NoteMind AI")

# Sidebar
with st.sidebar:
    st.header("1. Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    st.header("2. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Load free local embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Main App logic
if uploaded_file and groq_api_key:
    temp_pdf_path = f"./temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        os.remove(temp_pdf_path)

    st.success("PDF Loaded Successfully!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching PDF..."):
                llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
                
                system_prompt = (
                    "You are a helpful assistant. Use the following context to answer the question.\n\n"
                    "{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

elif not groq_api_key:
    st.info("👈 Please enter your Groq API key in the left sidebar.")
elif not uploaded_file:
    st.info("👈 Please upload a PDF file to start.")