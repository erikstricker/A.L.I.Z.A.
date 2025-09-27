import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
import os
from ui_components import HomePageUI

# Load environment variables from .env file
load_dotenv()

import yaml
from yaml.loader import SafeLoader
with open('../config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

st.set_page_config(
    page_title="LLM Bootcamp Project",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# 1.RENDER THE LOGIN WIDGET
authenticator.login('Login', 'main')

# 2. ADD LOGIC TO DISPLAY THE PAGE CONTENT OR MESSAGES
if st.session_state["authentication_status"]:
    # Display content for logged-in users
    authenticator.logout('Logout', 'sidebar') #  Add a logout button to the sidebar
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')

    # Apply centralized home page styling
    HomePageUI.apply_home_styling()

    # Render hero section using centralized component
    HomePageUI.render_hero_section()

    st.markdown("### Available AI Assistants:")

    # Explicitly defined page configurations
    pages = [
        {"icon": "💬", "title": "Basic AI Chat", "description": "Simple AI conversation", "file": "pages/1_basic.py"},
        {"icon": "🔍", "title": "Search Enabled Chat", "description": "AI with internet search capabilities", "file": "pages/2_chatbot_agent.py"},
        {"icon": "📚", "title": "RAG", "description": "Retrieval-Augmented Generation with documents", "file": "pages/3_chat_with_your_data.py"},
        {"icon": "🔧", "title": "MCP Chatbot", "description": "Model Context Protocol integration", "file": "pages/4_mcp_agent.py"},
    ]

    # Enhanced feature list using centralized components
    st.markdown("""
    <div class="feature-list">
        <h3 style="color: #00d4aa; margin-bottom: 1.5rem; text-align: center;">✨ Available Features</h3>
    </div>
    """, unsafe_allow_html=True)

    for page_info in pages:
        HomePageUI.render_feature_card(page_info["icon"], page_info["title"], page_info["description"])

elif st.session_state["authentication_status"] is False:
    # ---> Display message for incorrect credentials
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    # ---> Display message for users who haven't logged in yet
    st.warning('Please enter your username and password')