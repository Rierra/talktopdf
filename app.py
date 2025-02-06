'''import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq  # Use Groq's LLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # Use HuggingFace or Groq's embedding model
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # Use Groq's model here
    llm = ChatGroq(model_name="mixtral-8x7b-32768", api_key="your_groq_api_key")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Define your templates
    user_template = "<div style='color: blue;'>User: {{MSG}}</div>"
    bot_template = "<div style='color: green;'>Bot: {{MSG}}</div>"

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()'''

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
st.set_page_config(page_icon="💡", layout="wide", page_title="LiMO")

# Sidebar for model selection and PDF upload
with st.sidebar:
    st.title('🤗💬 LiMO')
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
st.subheader("LiMO, Light finance's helper", divider="rainbow", anchor=False)

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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
