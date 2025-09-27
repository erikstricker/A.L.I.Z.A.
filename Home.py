import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LLM Bootcamp Project",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- USER AUTHENTICATION ---
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- LOGIN WIDGET ---
# The login widget is rendered in the main body of the page.
authenticator.login()

if st.session_state["authentication_status"]:
    # --- LOGGED IN WIDGETS ---
    # Show a logout button in the sidebar once the user is logged in.
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')

    # --- MAIN PAGE CONTENT ---
    st.title("Welcome to the LLM Bootcamp Project! 👋")
    st.markdown("This is the central hub for all available AI assistants. Please select a feature from the list below or from the sidebar to get started.")

    st.markdown("### Available AI Assistants:")

    # Example of how you can display your pages/features
    st.info("Navigate to the protected pages using the sidebar on the left.")


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password to access the application.')