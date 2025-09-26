"""Basic Chatbot Page.

A modern, conversational AI chatbot with customizable response styles
and conversation memory. Features a clean interface with enhanced styling
and reliable message processing.
"""

import streamlit as st
from typing import Dict, Any, List

from ui_components import ChatbotUI, APIKeyUI
from langchain_helpers import BasicChatbotHelper, ValidationHelper


def configure_api_key() -> bool:
    """Configure OpenAI API key for the chatbot.
    
    Handles API key collection, validation, and session state management.
    Uses centralized UI components for consistent styling.
    
    Returns:
        True if API key is configured and valid, False otherwise
    """
    api_key = st.session_state.get("basic_openai_key", "")
    
    if not api_key:
        # Check if we just connected (avoid showing form again)
        if st.session_state.get("basic_api_key_connected", False):
            st.session_state["basic_api_key_connected"] = False
            return True
        
        # Use centralized API key form
        inputs = APIKeyUI.render_api_key_form(
            title="🔑 Enter API Key",
            inputs=[{
                "key": "basic_api_key_input",
                "label": "OpenAI API Key",
                "placeholder": "sk-proj-...",
                "password": True
            }]
        )
        
        if inputs:
            api_key_input = inputs.get("basic_api_key_input", "")
            if ValidationHelper.validate_openai_key(api_key_input):
                st.session_state["basic_openai_key"] = api_key_input
                st.session_state["basic_api_key_connected"] = True
                st.rerun()
            else:
                st.error("❌ Invalid key format")
        return False
    
    return True

def display_messages() -> None:
    """Display chat messages using centralized UI components.
    
    Shows conversation history or welcome message if no messages exist.
    Uses the ChatbotUI component for consistent message rendering.
    """
    if not ChatbotUI.display_chat_messages(st.session_state.basic_messages):
        st.info("🤖 Ask me anything and I'll be happy to help!")

def main() -> None:
    """Main application function.
    
    Orchestrates the entire chatbot page including:
    - Page setup and styling
    - API key configuration
    - Chat interface and message processing
    - Response generation with error handling
    """
    # Use centralized UI setup
    ChatbotUI.setup_page("AI Chat", "🚀")
    
    # Use centralized header component
    ChatbotUI.render_page_header(
        "🚀", 
        "Basic Chatbot", 
        "AI conversation assistant with memory"
    )
    
    # Validate API key configuration before proceeding
    if not configure_api_key():
        return
    
    
    # Initialize chat interface and processing logic
    with st.container():
        
        # Configure LangChain with default chatbot settings
        config = BasicChatbotHelper.get_default_config()
        api_key = st.session_state.get("basic_openai_key", "")
        
        # Ensure chain is properly initialized with current API key
        if api_key and ("basic_chain" not in st.session_state or 
                       st.session_state.get("basic_current_api_key") != api_key):
            st.session_state.basic_chain = BasicChatbotHelper.build_chain(config, api_key)
            st.session_state.basic_current_api_key = api_key
        elif not api_key:
            st.error("API key not found. Please refresh the page.")
            return
        
        # Initialize conversation history in session state
        if "basic_messages" not in st.session_state:
            st.session_state.basic_messages = []
        
        # Render current conversation history
        display_messages()
        
        # Process pending user message and generate AI response
        if (st.session_state.basic_messages and 
            st.session_state.basic_messages[-1]["role"] == "user" and
            not st.session_state.get("basic_processing", False)):
            
            st.session_state.basic_processing = True
            try:
                # Show processing indicator
                with st.chat_message("assistant", avatar=ChatbotUI.get_bot_avatar()):
                    with st.spinner("Thinking..."):
                        # Get the last user message
                        user_input = st.session_state.basic_messages[-1]["content"]
                        response = BasicChatbotHelper.invoke_with_memory(
                            st.session_state.basic_chain, 
                            user_input, 
                            st.session_state.basic_messages
                        )
                        
                        # Add assistant response
                        st.session_state.basic_messages.append({
                            "role": "assistant", 
                            "content": response.content
                        })
                
                st.session_state.basic_processing = False
                st.rerun()
                
            except Exception as e:
                st.session_state.basic_processing = False
                st.error(f"Error: {str(e)}")
                st.rerun()

    # Chat input - outside container to prevent shifting
    if prompt := st.chat_input("Type your message here..."):
        # Add user message and rerun to show it first
        st.session_state.basic_messages.append({"role": "user", "content": prompt})
        st.rerun()

if __name__ == "__main__":
    main()