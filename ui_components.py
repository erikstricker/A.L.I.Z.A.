"""UI Components and Styling Module.

Centralized UI styling and components for the LLM Bootcamp Project.
Provides reusable UI components and consistent styling across all pages:
- ChatbotUI: Core chat interface components and styling
- HomePageUI: Home page specific components and layouts
- APIKeyUI: API key configuration forms and validation
"""

import streamlit as st

class ChatbotUI:
    """Centralized UI components and styling for chatbot pages.
    
    Provides consistent chat interface components, page setup utilities,
    and enhanced dark theme styling for all chatbot pages.
    """
    
    # Avatar emojis for consistent chat message display (non-copyrighted)
    USER_AVATAR = "👤"
    BOT_AVATAR = "🤖"
    
    @staticmethod
    def get_large_emoji_avatar(emoji: str, size: int = 64) -> str:
        """Create a larger emoji avatar using data URI.
        
        Args:
            emoji: The emoji character to display
            size: Size of the emoji in pixels
            
        Returns:
            Data URI string for the emoji image
        """
        import base64
        
        # Create SVG with larger emoji
        svg_content = f'''
        <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <text x="50%" y="50%" font-size="{size-10}" text-anchor="middle" dominant-baseline="central" font-family="Apple Color Emoji, Segoe UI Emoji, Noto Color Emoji, sans-serif">
                {emoji}
            </text>
        </svg>
        '''
        
        # Encode as data URI
        encoded = base64.b64encode(svg_content.encode()).decode()
        return f"data:image/svg+xml;base64,{encoded}"
    
    @staticmethod
    def get_user_avatar() -> str:
        """Get larger user avatar."""
        return ChatbotUI.get_large_emoji_avatar(ChatbotUI.USER_AVATAR)
    
    @staticmethod
    def get_bot_avatar() -> str:
        """Get larger bot avatar."""
        return ChatbotUI.get_large_emoji_avatar(ChatbotUI.BOT_AVATAR)
    
    @staticmethod
    def apply_enhanced_styling() -> None:
        """Apply enhanced dark theme styling to the page.
        
        Injects custom CSS for a modern, dark-themed interface with
        gradient effects, enhanced buttons, and improved visual hierarchy.
        """
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
            
            /* Enhanced titles */
            h1 {
                background: linear-gradient(135deg, #00d4aa, #ffffff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                text-shadow: 0 0 30px rgba(0, 212, 170, 0.5);
            }
            
            /* Chat input enhancement */
            .stChatInput {
                background: linear-gradient(135deg, #1e1e2e, #2a2a3a);
                border-radius: 15px;
                border: 2px solid rgba(0, 212, 170, 0.2);
            }
            
            /* Info boxes */
            .stInfo {
                background: linear-gradient(135deg, #1e1e2e, #2a2a3a);
                border-left: 4px solid #00d4aa;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 212, 170, 0.2);
            }
            
            /* Success boxes */
            .stSuccess {
                background: linear-gradient(135deg, #1e1e2e, #2a2a3a);
                border-left: 4px solid #00d4aa;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 212, 170, 0.2);
            }
            
            /* Error boxes */
            .stError {
                background: linear-gradient(135deg, #2e1e1e, #3a2a2a);
                border-left: 4px solid #ff6b6b;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(255, 107, 107, 0.2);
            }
            
            /* Warning boxes */
            .stWarning {
                background: linear-gradient(135deg, #2e2e1e, #3a3a2a);
                border-left: 4px solid #ffaa00;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(255, 170, 0, 0.2);
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
            
            /* Enhanced progress bars */
            .stProgress > div > div {
                background: linear-gradient(135deg, #00d4aa, #00a883);
            }
            
            /* Loading spinner enhancement */
            .stSpinner {
                color: #00d4aa;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def setup_page(title: str, icon: str, layout: str = "wide", sidebar_state: str = "collapsed") -> None:
        """Setup page configuration with enhanced styling.
        
        Args:
            title: Page title for browser tab
            icon: Page icon emoji or image
            layout: Streamlit layout mode ('wide' or 'centered')
            sidebar_state: Initial sidebar state ('collapsed' or 'expanded')
        """
        st.set_page_config(
            page_title=title,
            page_icon=icon,
            layout=layout,
            initial_sidebar_state=sidebar_state
        )
        ChatbotUI.apply_enhanced_styling()
    
    @staticmethod
    def render_page_header(icon: str, title: str, subtitle: str) -> None:
        """Render enhanced page header with title and subtitle.
        
        Creates a centered, styled header with gradient effects and
        proper typography hierarchy.
        
        Args:
            icon: Header icon emoji or image
            title: Main page title
            subtitle: Descriptive subtitle text
        """
        st.markdown(f"""
        <div style='text-align: center; margin: 0.5rem 0 1rem 0;'>
            <h1 style='font-size: 2.625rem; margin-bottom: 1rem; text-shadow: 0 0 30px rgba(0, 212, 170, 0.5);'>
                {icon} {title}
            </h1>
            <p style='font-size: 0.9rem; color: #a0a0a0; margin-top: -0.5rem;'>
                {subtitle}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_chat_message(role: str, content: str, avatar_url: str = None) -> None:
        """Render chat message with proper avatar.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content text
            avatar_url: Optional custom avatar URL (uses defaults if None)
        """
        if avatar_url is None:
            avatar_url = ChatbotUI.get_user_avatar() if role == "user" else ChatbotUI.get_bot_avatar()
        
        with st.chat_message(role, avatar=avatar_url):
            st.write(content)
    
    @staticmethod
    def render_processing_message(message: str = "Processing...", avatar_url: str = None):
        """Render processing message with spinner.
        
        Args:
            message: Processing status message to display
            avatar_url: Optional custom avatar URL for the processing message
            
        Yields:
            Context manager for the processing spinner
        """
        if avatar_url is None:
            avatar_url = ChatbotUI.BOT_AVATAR
        
        with st.chat_message("assistant", avatar=avatar_url):
            with st.spinner(message):
                yield
    
    @staticmethod
    def display_chat_messages(messages: list) -> bool:
        """Display list of chat messages with proper avatars.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            True if messages were displayed, False if no messages to display
        """
        if not messages:
            return False
        
        for message in messages:
            ChatbotUI.render_chat_message(
                message["role"], 
                message["content"]
            )
        return True


class HomePageUI:
    """UI components specific to the home page.
    
    Provides specialized components for the landing page including
    hero sections, feature cards, and navigation buttons.
    """
    
    @staticmethod
    def apply_home_styling() -> None:
        """Apply enhanced styling specific to the home page.
        
        Adds custom CSS for enhanced cards, navigation buttons,
        and feature list styling optimized for the landing page.
        """
        st.markdown("""
        <style>
            /* Enhanced card styling */
            .stButton > button {
                background: linear-gradient(135deg, #00d4aa, #00a883);
                border: none;
                border-radius: 15px;
                color: white;
                font-weight: 600;
                padding: 1rem 2rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 20px rgba(0, 212, 170, 0.3);
                font-size: 1.1rem;
                height: 80px;
            }
            
            .stButton > button:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0, 212, 170, 0.4);
                background: linear-gradient(135deg, #00e6c0, #00cc99);
            }
            
            /* Enhanced main title */
            .main-title {
                background: linear-gradient(135deg, #00d4aa, #ffffff, #00d4aa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                text-shadow: 0 0 40px rgba(0, 212, 170, 0.5);
            }
            
            /* Feature list styling */
            .feature-list {
                background: linear-gradient(135deg, #1e1e2e, #2a2a3a);
                padding: 2rem;
                border-radius: 15px;
                border: 1px solid rgba(0, 212, 170, 0.1);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
                margin-top: 2rem;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_hero_section() -> None:
        """Render the hero section for home page.
        
        Creates the main landing page header with gradient title,
        descriptive subtitle, and proper spacing.
        """
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0 2rem 0;">
            <h1 class="main-title" style="font-size: 3rem; margin-bottom: 1rem;">
                🤖 LLM Bootcamp Project
            </h1>
            <p style="font-size: 1.125rem; color: #a0a0a0; margin-bottom: 2rem;">
                Explore AI chatbot with multiple capabilities
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_feature_card(icon: str, title: str, description: str) -> None:
        """Render individual feature card.
        
        Args:
            icon: Feature icon emoji or image
            title: Feature title text
            description: Feature description text
        """
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1e1e2e, #2a2a3a);
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 10px;
            border-left: 4px solid #00d4aa;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        ">
            <strong style="color: #00d4aa;">{icon} {title}</strong> - <span style="color: #cccccc;">{description}</span>
        </div>
        """, unsafe_allow_html=True)


class APIKeyUI:
    """UI components for API key configuration.
    
    Provides reusable forms and input components for collecting
    and validating API keys and configuration settings.
    """
    
    @staticmethod
    def render_api_key_form(title: str = "🔑 Enter API Key", inputs: list = None) -> dict:
        """Render centered API key configuration form.
        
        Creates a responsive, centered form for collecting API keys
        and other configuration inputs with validation.
        
        Args:
            title: Form title to display
            inputs: List of input configuration dictionaries
            
        Returns:
            Dictionary of input values if form submitted, None otherwise
        """
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"### {title}")
            
            # Render configured input fields
            input_values = {}
            if inputs:
                for input_config in inputs:
                    input_values[input_config["key"]] = st.text_input(
                        input_config["label"],
                        type="password" if input_config.get("password", True) else "text",
                        placeholder=input_config.get("placeholder", ""),
                        value=input_config.get("value", ""),
                        key=input_config["key"]
                    )
            
            # Submit button with full-width styling
            if st.button("Connect", type="primary", use_container_width=True):
                return input_values
        
        return None