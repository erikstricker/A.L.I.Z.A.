"""
Chat with your Data Page.

RAG (Retrieval-Augmented Generation) powered chatbot that enables
intelligent question-answering over user-uploaded PDF documents.
This script manages the UI and calls RAGHelper for backend processing.
"""
import streamlit as st
import google.generativeai as genai
from google.api_core import exceptions
import time
import yaml
from yaml.loader import SafeLoader
from typing import List, Any

# Authentication and UI
import streamlit_authenticator as stauth
from ui_components import ChatbotUI # Assuming this is in your project

# --- RAG-RELATED CHANGE ---
# Import your helper class and remove the other now-redundant LangChain imports.
from langchain_helpers import RAGHelper

# --- Streamlit Page Setup & UI --- (No changes below this line)

def setup_page() -> None:
    """Sets up the Streamlit page with configuration and custom CSS."""
    st.set_page_config(
        page_title="Chat with Documents", 
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown("""<style> ... </style>""", unsafe_allow_html=True)

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
    
    # --- RAG-RELATED CHANGE ---
    # The __init__ method is simplified. It no longer needs to create
    # the LLM or embeddings models itself.
    def __init__(self) -> None:
        """Initializes the chatbot UI handler."""
        pass

    # --- RAG-RELATED CHANGE ---
    # This method now correctly gets the Gemini API key from the session state
    # and passes it to your RAGHelper.
    def setup_graph(self, uploaded_files: List[Any]) -> Any:
        """Setup RAG processing graph by calling the RAGHelper."""
        api_key = st.session_state.get("gemini_api_key", "")
        return RAGHelper.setup_rag_system(uploaded_files, api_key)


    def display_messages(self) -> None:
        """Display document-aware chat messages."""
        if "rag_messages" in st.session_state:
            for message in st.session_state.rag_messages:
                if message["role"] == "user":
                    with st.chat_message("user", avatar=ChatbotUI.get_user_avatar()):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant", avatar=ChatbotUI.get_bot_avatar()):
                        st.write(message["content"])

    def main(self) -> None:
        """Main RAG chatbot workflow."""
        if "rag_uploaded_files" not in st.session_state:
            st.session_state.rag_uploaded_files = []
        if "rag_app" not in st.session_state:
            st.session_state.rag_app = None
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        col1, col2, col3 = st.columns([2, 1.5, 2])
        with col2:
            uploaded_files = st.file_uploader(
                label="**Upload PDF files to chat with your documents**",
                type=["pdf"],
                accept_multiple_files=True
            )
        st.markdown("<br>", unsafe_allow_html=True)

        if uploaded_files:
            current_files = {f.name for f in uploaded_files}
            previous_files = {f.name for f in st.session_state.get("rag_uploaded_files", [])}
            
            if current_files != previous_files or st.session_state.rag_app is None:
                st.session_state.rag_uploaded_files = uploaded_files
                with st.spinner("📚 Processing documents..."):
                    st.session_state.rag_app = self.setup_graph(uploaded_files)
        
        self.display_messages()
        
        # This section handles invoking the RAG chain created by your RAGHelper
        if (st.session_state.get("rag_app") and 
            st.session_state.rag_messages and 
            st.session_state.rag_messages[-1]["role"] == "user" and
            not st.session_state.get("rag_processing", False)):
            
            st.session_state.rag_processing = True
            try:
                with st.chat_message("assistant", avatar=ChatbotUI.get_bot_avatar()):
                    with st.spinner("Analyzing documents..."):
                        user_query = st.session_state.rag_messages[-1]["content"]
                        
                        # The invoke call now works with the graph from RAGHelper
                        result = st.session_state.rag_app.invoke(
                            {"question": user_query}
                        )
                        
                        answer = result.get("generation", "I couldn't find an answer in the documents.")
                        
                        st.session_state.rag_messages.append({"role": "assistant", "content": answer})
                
                st.session_state.rag_processing = False
                st.rerun()
                
            except Exception as e:
                st.session_state.rag_processing = False
                st.error(f"Error: {str(e)}")
                st.rerun()

        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            st.rerun()

def main() -> None:
    """Main application function."""
    setup_page()
    
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
    """, unsafe_allow_html=True)
    
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")

    if not configure_api_key(api_key=gemini_api_key):
        st.stop()
    
    # --- RAG-RELATED CHANGE ---
    # Simplified the app initialization
    app = CustomDataChatbot()
    app.main()

if __name__ == "__main__":
    main()