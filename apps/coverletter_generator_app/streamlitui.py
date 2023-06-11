from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import json
import uuid
import openai
import os
import tempfile
from coverletter_generator import ResumeExtractor, CoverLetterGenerator

st.set_page_config(page_title="Cover Letter Generator - Upload a 1-Page Resume and Get Back a Cover Letter.")

def display_messages():
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def read_and_save_file():
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            resume_extractor = ResumeExtractor(file_path)
            resume_details = json.loads(resume_extractor.extract_details())
            if resume_details["is_resume"]:
                coverletter_generator = CoverLetterGenerator(resume_details)
                st.session_state["messages"].append((coverletter_generator.get_coverletter(), False))
            else:
                st.session_state["messages"].append(("The pdf uploaded wasn't that of a Resume.",False))

        try:
            os.remove(file_path)
        except Exception as e:
            print(e)

def is_openai_api_key_set() -> bool:
    return len(st.session_state["OPENAI_API_KEY"]) > 0

def run() -> None:
    if len(st.session_state) == 0:
        load_dotenv()
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 

    st.header("Cover Letter Generator: Upload a 1-Page Resume, And Get A Cover Letter.")

    st.subheader("Upload a Resume")
    st.file_uploader(
        "Upload a Resume",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
        disabled=len(st.session_state["OPENAI_API_KEY"]) <= 0)

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
    st.divider()
    st.text("Made by Moko.")

if __name__ == '__main__':
    run()
    