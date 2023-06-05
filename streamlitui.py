import os
import tempfile
import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFium2Loader 
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

OPENAI_API_KEY = 'enter your open ai key'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class ResumeExtractor(object):
    def __init__(self, path : str) -> None:

        # Precondition checks.
        if path == "" or path == None:
            raise ValueError(f"The path provided: {path} is not a valid path.")
        #if not path.endswith(".pdf"): raise ValueError(f"The path provided: {path} is not a valid pdf path.") 

        self.path = path
        loader = PyPDFium2Loader(path)
        self.pages  = loader.load_and_split()
        embeddings = OpenAIEmbeddings()
        self.docsearch = Chroma.from_documents([self.pages[0]], embeddings).as_retriever(search_kwargs={ "k": 1 })

    def ask(self, question : str) -> str:
        docs = self.docsearch.get_relevant_documents(question)
        chain = load_qa_chain(OpenAI(temperature=0, max_tokens=3000), chain_type="stuff")
        output = chain.run(input_documents=docs, question=question)
        return output

    def extract_details(self) -> str:
        query_to_extract_info = """Using the document, answer the following questions and output valid json with property names enclosed with double quotes with keys: "is_resume", "skills", "years_of_experience", "experience_summary", "achievements", "highest_education", "specialization":

        1. Is this document of a resume? Answer in "True" or "False". The answer should correspond to the "is_resume key".
        2. What are the candidates skills? The answer should be a json list associated with the "skills" key.
        3. How many years of experience does the candidate have? The answer should correspond to the "years_of_experience" key.
        4. Based on the candidate's experience, extract achievements that are backed by numbers that the candidate has made in the form of a json list associated with the "achievements" key.
        5. What is the candidate's highest education? The answer should either be: High School, Bachelors, Masters, PhD or specify NA if you don't know. This answer should correspond to the "highest_education" key.
        6. What is the candidate's major or field of study? The answer should correspond to the "specialization" key."""
        return self.ask(query_to_extract_info)

st.set_page_config(page_title="ChatPDF")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            query_text = st.session_state["resume_extractor"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((query_text, False))

def read_and_save_file():
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["resume_extractor"] = ResumeExtractor(file_path)
        os.remove(file_path)

def is_openai_api_key_set() -> bool:
    return len(st.session_state["OPENAI_API_KEY"]) > 0

def run() -> None:
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

    st.header("ChatPDF")

    if st.text_input("OpenAI API Key", value=st.session_state["OPENAI_API_KEY"], key="input_OPENAI_API_KEY", type="password"):
        if (
            len(st.session_state["input_OPENAI_API_KEY"]) > 0
            and st.session_state["input_OPENAI_API_KEY"] != st.session_state["OPENAI_API_KEY"]
        ):
            st.session_state["OPENAI_API_KEY"] = st.session_state["input_OPENAI_API_KEY"]
            st.session_state["messages"] = []
            st.session_state["user_input"] = ""

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
        disabled=not is_openai_api_key_set(),
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Job Description", key="user_input", disabled=not is_openai_api_key_set(), on_change=process_input)

    st.divider()

if __name__ == '__main__':
    run()
    