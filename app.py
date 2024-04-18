import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain
from langchain.memory import StreamlitChatMessageHistory
from utils import save_chat_history_json, load_chat_history_json, get_timestamp
from pdf_handler import add_documents_to_db
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Functions for loading the LLM chain, managing chat interface and pdf chatting
def load_chain(chat_history):
    if st.session_state.pdf_chat:
        return load_pdf_chat_chain(chat_history)
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

            
def toggle_pdf_chat():
    st.session_state.pdf_chat = True

def main():
    st.title("Chat with PDF APP")
    chat_container = st.container()

    # initialisation of session state variables
    if "send_input" not in st.session_state:
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None

    # sidebar
    st.sidebar.toggle("PDF chat", key="pdf_chat", value = False)

    # chat history and llm chain initialisation
    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Enter your message", key="user_input", on_change=set_send_input)

    send_button = st.button("Send",key="send_button")

    # PDF upload at sidebar
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", accept_multiple_files = True, key = "pdf_upload", type=["pdf"], on_change=toggle_pdf_chat)

    if uploaded_pdf:
        with st.spinner("Extracting text from PDFs"):
            add_documents_to_db(uploaded_pdf)

    # Idle state for chat, the state gets updated when the first user messsage is sent.
    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":

    
            with chat_container:
                st.chat_message("User").write(st.session_state.user_question)
                llm_response = llm_chain.run(st.session_state.user_question)
                st.chat_message("LLM").write(llm_response)
                st.session_state.user_question = ""

    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)


if __name__ == "__main__":
    main()