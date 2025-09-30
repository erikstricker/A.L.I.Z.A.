"""
Chat with your Data Page.

RAG (Retrieval-Augmented Generation) powered chatbot that enables
intelligent question-answering over user-uploaded PDF documents.
This version is fully integrated with Google Gemini for both embeddings and chat.
"""
import streamlit as st
import google.generativeai as genai
from google.api_core import exceptions
import time
import yaml
from yaml.loader import SafeLoader
from typing import List, Any

# LangChain components for RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Authentication and UI
import streamlit_authenticator as stauth
from ui_components import ChatbotUI # Assuming this is in your project

# --- RAG Core Logic ---
# This class replaces the need for an external 'langchain_helpers.py' file

class RAGSystem:
    """Encapsulates the entire RAG pipeline."""
    
    def __init__(self, api_key: str):
        """Initializes the RAG system with the user's API key."""
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            google_api_key=self.api_key
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )

    def build_vector_store(self, uploaded_files: list):
        """
        Builds a FAISS vector store from uploaded PDF files using batching to handle rate limits.
        """
        # 1. Load and split the documents into chunks
        all_docs = []
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(file.name)
            all_docs.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_docs)
        
        if not chunks:
            st.error("Could not extract any text chunks from the PDF.")
            return None

        # 2. Process chunks in batches to respect API rate limits
        batch_size = 50  # Number of chunks to process at a time
        vector_store = None
        
        for i in range(0, len(chunks), batch_size):
            # Get the current batch of chunks
            batch = chunks[i:i + batch_size]
            
            st.write(f"Embedding batch {i//batch_size + 1}/{(len(chunks) - 1)//batch_size + 1}...")

            if vector_store is None:
                # Create the vector store with the first batch
                vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                # Add subsequent batches to the existing store
                vector_store.add_documents(batch)
                
            # Pause to avoid hitting the rate limit
            time.sleep(5) # Wait for 5 seconds between batches

        return vector_store

    def create_rag_chain(self, vector_store):
        """Creates a runnable RAG chain."""
        retriever = vector_store.as_retriever()
        
        template = """
        You are a helpful assistant. Answer the question based only on the following context.
        If you cannot answer the question from the context, please state that you cannot find the answer in the provided documents.

        CONTEXT:
        {context}

        QUESTION:
        {question}
        """
        
        prompt = PromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

# --- Streamlit Page Setup & UI ---

def setup_page() -> None:
    """Sets up the Streamlit page with configuration and custom CSS."""
    st.set_page_config(
        page_title="Chat with Documents", 
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    # Your CSS markdown here (omitted for brevity, but you should keep yours)
    st.markdown("""<style> ... </style>""", unsafe_allow_html=True) # Keep your CSS

def validate_gemini_key(api_key: str) -> bool:
    """Checks if a provided Google Gemini API key is valid."""
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        genai.list_models()
        return True
    except exceptions.PermissionDenied:
        st.error("API Key is invalid or has insufficient permissions.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during API key validation: {e}")
        return False

def configure_api_key(api_key: str = None) -> bool:
    """Configures the Gemini API key via direct input or a Streamlit UI."""
    if api_key and validate_gemini_key(api_key):
        st.session_state["gemini_api_key"] = api_key
        return True
    if "gemini_api_key" in st.session_state and validate_gemini_key(st.session_state["gemini_api_key"]):
        return True
    
    st.markdown("### 🔑 Enter your Gemini API Key")
    st.markdown("You can get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey).")
    
    with st.form("gemini_api_key_form"):
        user_api_key = st.text_input(
            "Gemini API Key", type="password", placeholder="Enter your key here", label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Connect", type="primary")
        if submitted and validate_gemini_key(user_api_key):
            st.session_state["gemini_api_key"] = user_api_key
            st.success("✅ API Key connected successfully!")
            time.sleep(1)
            st.rerun()
    return False

# --- Main Application Class ---

class CustomDataChatbot:
    """Manages the Streamlit UI and workflow for the RAG chatbot."""
    
    def __init__(self, api_key: str) -> None:
        """Initializes the chatbot UI handler with the Gemini API key."""
        self.api_key = api_key

    def display_messages(self) -> None:
        """Displays chat messages from the session state."""
        if "rag_messages" in st.session_state:
            for message in st.session_state.rag_messages:
                avatar = ChatbotUI.get_user_avatar() if message["role"] == "user" else ChatbotUI.get_bot_avatar()
                with st.chat_message(message["role"], avatar=avatar):
                    st.write(message["content"])

    def main(self) -> None:
        """Main RAG chatbot workflow."""
        # Initialize session state variables
        if "rag_uploaded_files" not in st.session_state:
            st.session_state.rag_uploaded_files = []
        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = None
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        # Document uploader
        col1, col2, col3 = st.columns([2, 1.5, 2])
        with col2:
            uploaded_files = st.file_uploader(
                label="**Upload PDF files to chat with your documents**",
                type=["pdf"],
                accept_multiple_files=True
            )
        st.markdown("<br>", unsafe_allow_html=True)

        # Process documents if they change
        if uploaded_files:
            current_files = {f.name for f in uploaded_files}
            previous_files = {f.name for f in st.session_state.get("rag_uploaded_files", [])}
            
            if current_files != previous_files or st.session_state.rag_chain is None:
                st.session_state.rag_uploaded_files = uploaded_files
                with st.spinner("📚 Processing documents... This may take a moment."):
                    rag_system = RAGSystem(self.api_key)
                    vector_store = rag_system.build_vector_store(uploaded_files)
                    st.session_state.rag_chain = rag_system.create_rag_chain(vector_store)
                    st.session_state.rag_messages = [] # Clear chat on new docs
                    st.success("Documents processed successfully! You can now ask questions.")
        
        # Display chat history and handle new queries
        self.display_messages()
        
        if st.session_state.rag_chain:
            if prompt := st.chat_input("Ask about your documents..."):
                st.session_state.rag_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar=ChatbotUI.get_user_avatar()):
                    st.write(prompt)

                with st.chat_message("assistant", avatar=ChatbotUI.get_bot_avatar()):
                    with st.spinner("Analyzing documents..."):
                        try:
                            response = st.session_state.rag_chain.invoke(prompt)
                            st.write(response)
                            st.session_state.rag_messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_message = f"An error occurred: {e}"
                            st.error(error_message)
                            st.session_state.rag_messages.append({"role": "assistant", "content": error_message})
        elif uploaded_files:
             st.info("Still processing documents, please wait...")
        else:
             st.info("Please upload PDF documents to begin the chat.")


def main() -> None:
    """Main application function."""
    setup_page() # Use your original setup_page function for styling
    
    # --- AUTHENTICATION ---
    with open('./config.yaml') as file:
        config_data = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config_data['credentials'],
        config_data['cookie']['name'],
        config_data['cookie']['key'],
        config_data['cookie']['expiry_days']
    )

    if not st.session_state.get("authentication_status"):
        st.warning("Please log in to access this page.")
        st.stop()
    
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')
    # --- END OF AUTHENTICATION ---
    
    st.markdown("""
    <div style='text-align: center; margin: 0.5rem 0 1rem 0;'>
        <h1 style='...'>📚 Chat with your Data</h1>
        <p style='...'>Upload documents and get intelligent answers using RAG</p>
    </div>
    """, unsafe_allow_html=True) # Keep your HTML/CSS
    
    # Use st.secrets for the API key in a deployed app
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")

    if not configure_api_key(api_key=gemini_api_key):
        st.stop()
    
    # Initialize and run the chatbot
    app = CustomDataChatbot(api_key=st.session_state["gemini_api_key"])
    app.main()

if __name__ == "__main__":
    main()