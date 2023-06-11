from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
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
from resume_comparer import ResumeExtractor, JobDescriptionExtractor, ResumeComparer

st.set_page_config(page_title="Resume Chain - Upload a 1-Page Resume, Add a Job Description and Get Details.")

def display_messages():
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"]
        with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix=".txt") as tf:
            tf.write(str(user_text))
        job_description_extractor = JobDescriptionExtractor(tf.name)
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            job_description_details = json.loads(job_description_extractor.extract_details())
            comparison = ResumeComparer(st.session_state["resume_details"], job_description_details)
            details = comparison.extract_details()

            #st.session_state["messages"].append((user_text, True))
            st.session_state["messages"].append((details["summary"], False))
            st.session_state["messages"].append((details["specifics"], False))
        try:
            os.remove(tf.name)
        except Exception as e:
            print(e)

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
            st.session_state["resume_details"] = resume_details 
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

    st.header("Resume Chain: Upload a 1-Page Resume, Add a Job Description and Get Details.")

    st.subheader("1. Upload a Resume")
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
    st.subheader("2. Enter a Job Description")
    st.text_area("Job Description", 
                  key="user_input", 
                  label_visibility="collapsed",
                  disabled=not is_openai_api_key_set(), 
                  on_change=process_input)

    st.divider()

if __name__ == '__main__':
    run()
    