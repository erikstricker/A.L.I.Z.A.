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

# --- MAIN PAGE CONTENT ---
st.title("👧 Welcome to A.L.I.Z.A.* 🤖")
st.markdown("This is the Atkinson lab AI chatbot, which let's chat with BCM wikis, lab 101s, and more.")


# --- LOGIN WIDGET ---
# The login widget is rendered in the main body of the page.
authenticator.login()

if st.session_state["authentication_status"]:
    # --- LOGGED IN WIDGETS ---
    # Show a logout button in the sidebar once the user is logged in.
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')

    # --- MAIN PAGE CONTENT ---
    st.title("A.L.I.Z.A. is ready for you!")

    st.markdown("### Available AI Assistants include:")

    # Example of how you can display your pages/features
    st.info("Navigate to the protected pages using the sidebar on the left.")


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password to access the application.')

st.markdown("*ELIZA was one of the first natural language processing chatbots developed from 1964 to 1967 by Joseph Weizenbaum at MIT.")
