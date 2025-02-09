import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from groq import Groq
import re

# Initialize Streamlit app
st.set_page_config(page_icon="📚", layout="wide", page_title="Riko: talk to pdf")

# Sidebar for model selection and PDF upload
with st.sidebar:
    st.title('Made with <3 Anmol')
    st.write("**Select Model & Settings**")
    
    models = {
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    }
    
    model_option = st.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"], index=0)
    max_tokens = st.slider("Max Tokens:", min_value=512, max_value=models[model_option]["tokens"], value=2048, step=512)
    
    st.write("**Upload a PDF to chat with**")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Display icon and title
st.subheader("Talk to pdf", divider="rainbow", anchor=False)

# Initialize Groq client
# Assuming you've set the environment variable 'GROQ_API_KEY' in your cloud environment
api_key = os.getenv("GROQ_API_KEY")

if api_key:
    client = Groq(api_key=api_key)
else:
    raise ValueError("API key not found!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Load and process uploaded PDF
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()
    
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)
    embedder = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embedder)
    st.session_state.vectorstore = vectorstore
    st.success("PDF processed successfully! You can start chatting.")

# Check if vectorstore is initialized
if st.session_state.vectorstore is not None:
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
else:
    retriever = None

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar='💡' if msg["role"] == "assistant" else '👨‍💻'):
        st.markdown(msg["content"])

# Function to clean response
def clean_response(response):
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

# Chat input
if retriever and (prompt := st.chat_input("How can I help you?")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)
    
    # Retrieve relevant PDF context
    retrieved_docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = f"Use only the following context to answer the question: {context}\nQuestion: {prompt}\nAnswer:"
    
    # Fetch response from Groq API
    try:
        response = client.chat.completions.create(
            model=model_option,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=max_tokens
        )
        bot_response = clean_response(response.choices[0].message.content)
    except Exception as e:
        bot_response = f"Error: {e}"
    
    # Display response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant", avatar="💡"):
        st.markdown(bot_response)

