"""Chat with your Data Page.

RAG (Retrieval-Augmented Generation) powered chatbot that enables
intelligent question-answering over user-uploaded PDF documents.
Features automatic document processing, vector search, and contextual responses.
"""

import streamlit as st
from typing import List, Dict, Any

from ui_components import ChatbotUI, APIKeyUI
from langchain_helpers import RAGHelper, ValidationHelper


def setup_page() -> None:
    """Set up the RAG page with enhanced styling.
    
    Configures page layout and applies custom CSS for document
    upload interface and chat components.
    """
    st.set_page_config(
        page_title="Chat with Documents", 
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced visual styling
    st.markdown("""
    <style>
        /* Enhanced chat styling */
        .stChatMessage {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 15px;
            background: linear-gradient(135deg, #1e1e2e, #2a2a3a);
            border: 1px solid rgba(0, 212, 170, 0.1);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        /* Enhanced buttons */
        .stButton > button {
            background: linear-gradient(135deg, #00d4aa, #00a883);
            border: none;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            padding: 0.75rem 2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
            background: linear-gradient(135deg, #00e6c0, #00cc99);
        }
        
        /* Enhanced text inputs */
        .stTextInput > div > div > input {
            background: linear-gradient(135deg, #1e1e2e, #2a2a3a);
            border: 2px solid rgba(0, 212, 170, 0.2);
            border-radius: 10px;
            color: #ffffff;
            font-size: 16px;
            padding: 12px;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #00d4aa;
            box-shadow: 0 0 20px rgba(0, 212, 170, 0.3);
        }
        
        /* Enhanced file uploader */
        .stFileUploader > div {
            background: linear-gradient(135deg, #1e1e2e, #2a2a3a);
            border: 2px dashed rgba(0, 212, 170, 0.3);
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stFileUploader > div:hover {
            border-color: #00d4aa;
            box-shadow: 0 0 20px rgba(0, 212, 170, 0.2);
        }
        
        /* Enhanced titles */
        h1 {
            background: linear-gradient(135deg, #00d4aa, #ffffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            text-shadow: 0 0 30px rgba(0, 212, 170, 0.5);
        }
        
        /* Enhanced progress bars */
        .stProgress > div > div {
            background: linear-gradient(135deg, #00d4aa, #00a883);
        }
    </style>
    """, unsafe_allow_html=True)
    

def configure_api_key() -> bool:
    """Configure OpenAI API key for RAG functionality.
    
    Handles API key collection and validation specifically
    for document processing and embedding generation.
    
    Returns:
        True if API key is configured and valid, False otherwise
    """
    api_key = st.session_state.get("rag_openai_key", "")
    
    if not api_key:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔑 Enter API Key")
            
            # Handle post-connection state to prevent form re-display
            if st.session_state.get("rag_key_connected", False):
                st.session_state["rag_key_connected"] = False
                return True
                
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-proj-...",
                key="rag_api_key_input"
            )
            
            if st.button("Connect", type="primary", use_container_width=True):
                if ValidationHelper.validate_openai_key(api_key_input):
                    st.session_state["rag_openai_key"] = api_key_input
                    st.session_state["rag_key_connected"] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid key format")
        return False
    
    return True

class CustomDataChatbot:
    """RAG-powered chatbot for document question answering.
    
    Processes uploaded PDF documents, creates vector embeddings,
    and enables intelligent querying with contextual responses.
    """
    
    def __init__(self) -> None:
        """Initialize the RAG chatbot with default settings."""
        self.openai_model = "gpt-4o-mini"

    def setup_graph(self, uploaded_files: List[Any]) -> Any:
        """Setup RAG processing graph from uploaded documents.
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            Configured RAG workflow for document Q&A
        """
        api_key = st.session_state.get("rag_openai_key", "")
        return RAGHelper.setup_rag_system(uploaded_files, api_key)
    
    def display_messages(self) -> None:
        """Display document-aware chat messages.
        
        Shows conversation history with document context awareness
        and helpful prompts for document-based queries.
        """
        if st.session_state.rag_messages:
            for message in st.session_state.rag_messages:
                if message["role"] == "user":
                    with st.chat_message("user", avatar=ChatbotUI.get_user_avatar()):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant", avatar=ChatbotUI.get_bot_avatar()):
                        st.write(message["content"])

    def main(self) -> None:
        """Main RAG chatbot workflow.
        
        Manages document upload, processing, vector store creation,
        and intelligent question-answering over document content.
        """
        # Initialize RAG-specific session state variables
        if "rag_uploaded_files" not in st.session_state:
            st.session_state.rag_uploaded_files = []
        if "rag_app" not in st.session_state:
            st.session_state.rag_app = None
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        # Centered document upload interface
        col1, col2, col3 = st.columns([2, 1.5, 2])
        with col2:
            uploaded_files = st.file_uploader(
                label="**Upload PDF files to chat with your documents**",
                type=["pdf"],
                accept_multiple_files=True
            )
            
            # Document processing handled automatically upon upload
                
        st.markdown("<br>", unsafe_allow_html=True)

        # Process documents when uploaded or changed
        if uploaded_files:
            current_files = {f.name for f in uploaded_files}
            previous_files = {f.name for f in st.session_state.get("rag_uploaded_files", [])}
            
            # Rebuild RAG system if files changed or system not initialized
            if current_files != previous_files or st.session_state.rag_app is None:
                st.session_state.rag_uploaded_files = uploaded_files
                with st.spinner("📚 Processing documents..."):
                    st.session_state.rag_app = self.setup_graph(uploaded_files)
        else:
            # Show welcome message when no documents are uploaded
            if not st.session_state.rag_messages:
                self.display_messages()
            return
            
        # Render conversation history with document context
        self.display_messages()
        
        # Process document-based query and generate contextual response
        if (st.session_state.rag_messages and 
            st.session_state.rag_messages[-1]["role"] == "user" and
            not st.session_state.get("rag_processing", False)):
            
            st.session_state.rag_processing = True
            try:
                # Show processing indicator
                with st.chat_message("assistant", avatar=ChatbotUI.get_bot_avatar()):
                    with st.spinner("Analyzing documents..."):
                        # Extract user query for document analysis
                        user_query = st.session_state.rag_messages[-1]["content"]
                        
                        # Process query through RAG workflow
                        result = st.session_state.rag_app.invoke({
                            "question": user_query, 
                            "mode": "fact", 
                            "documents": [], 
                            "generation": ""
                        })
                        
                        # Extract generated response with fallback
                        answer = (
                            result.get("generation", "").strip() or 
                            "I couldn't find enough information in the documents to answer that."
                        )
                        
                        # Add assistant response
                        st.session_state.rag_messages.append({"role": "assistant", "content": answer})
                
                st.session_state.rag_processing = False
                st.rerun()
                
            except Exception as e:
                st.session_state.rag_processing = False
                st.error(f"Error: {str(e)}")
                st.rerun()

        # Document query input interface
        if prompt := st.chat_input("Ask about your documents..."):
            # Add user query to conversation history
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            st.rerun()

def main() -> None:
    """Main application function for the RAG chatbot page.
    
    Orchestrates document upload, processing, and intelligent
    question-answering workflow with enhanced UI styling.
    """
    setup_page()
    
    # Page title - centered with enhanced styling
    st.markdown("""
    <div style='text-align: center; margin: 0.5rem 0 1rem 0;'>
        <h1 style='font-size: 2.625rem; margin-bottom: 1rem; text-shadow: 0 0 30px rgba(0, 212, 170, 0.5);'>
            📚 Chat with your Data
        </h1>
        <p style='font-size: 0.9rem; color: #a0a0a0; margin-top: -0.5rem;'>
            Upload documents and get intelligent answers using RAG
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Validate API key configuration for document processing
    if not configure_api_key():
        return
    
    # Initialize and run the document-aware chatbot
    app = CustomDataChatbot()
    app.main()

# Application entry point
if __name__ == "__main__":
    main()
