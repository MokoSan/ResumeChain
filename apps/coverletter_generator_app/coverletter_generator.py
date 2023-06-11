import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import json

class ResumeExtractor(object):
    def __init__(self, path : str) -> None:

        # Precondition checks.
        if path == "" or path == None:
            raise ValueError(f"The path provided: {path} is not a valid path.")

        self.path = path
        loader = UnstructuredPDFLoader(path)
        self.pages  = loader.load_and_split()
        if (len(self.pages) > 1):
            raise ValueError(f"The resume provided has more than 1 page. Please send a resume with just one page.")
        embeddings = OpenAIEmbeddings()

        # Just one page resumes accepted.
        self.docsearch = Chroma.from_documents([self.pages[0]], embeddings).as_retriever(search_kwargs={ "k": 1 })
        self.chain = load_qa_chain(OpenAI(temperature=0, max_tokens=2500), chain_type="stuff")

    def ask(self, question : str) -> str:
        docs = self.docsearch.get_relevant_documents(question)
        output = self.chain.run(input_documents=docs, question=question)
        return output

    def extract_details(self) -> str:
        query_to_extract_info = """Using the document, answer the following questions and output valid json with property names enclosed with double quotes with keys: "is_resume", "skills", "years_of_experience", "experience_summary", "achievements", "highest_education":

        1. Is this document of a resume? Answer in "True" or "False". The answer should correspond to the "is_resume" key.
        2. What are the candidates skills? The answer should be a json list associated with the "skills" key.
        3. How many years of experience does the candidate have? The answer should correspond to the "years_of_experience" key.
        4. Based on the candidate's experience, extract achievements that are backed by numbers that the candidate has made in the form of a json list associated with the "achievements" key.
        5. What is the candidate's highest education and field of study?"""
        return self.ask(query_to_extract_info)

class JobDescriptionExtractor(object):
    def __init__(self, path : str):
        # Precondition checks.
        if path == "" or path == None:
            raise ValueError(f"The path provided: {path} is not a valid path.")

        self.path = path
        self.loader = TextLoader(path, autodetect_encoding=True)
        #documents = self.loader.load()
        from langchain.indexes import VectorstoreIndexCreator
        self.index = VectorstoreIndexCreator().from_loaders([self.loader])

    def ask(self, question : str) -> str:
        output = self.index.query(question)
        return output

    def extract_details(self) -> str:
        query_to_extract_info = """Using the document, answer the following questions and output valid json with property names enclosed with double quotes with keys: "is_job_description", "skills_required", "responsibilities", "qualifications", "preferences":

        1. Is this document of a job description? Answer in "True" or "False". The answer should correspond to the "is_job_description" key.
        2. What are the skills required? The answer should be a json list associated with the "skills_required" key.
        3. What are the responsibilities? The answer should be a json list associated with the "responsibilities" key.
        4. What are the qualifications required? The answer should be in the form of a json list associated with the "qualifications" key.
        5. What are the preferences or preferred qualifications or skills? The answer should be in the form of a json list associated with the preferences key.
        """ 
        return self.ask(query_to_extract_info)

class ResumeComparer(object):
    def __init__(self, resume_details : json, job_description_details : json): 
        self.resume_details = resume_details
        self.job_description_details = job_description_details
        self.messages = [{"role": "system", "content" : "You are a sophisticated career advisor who is trying to discern if the resume data matches that of the job description."}]

    def extract_details(self) -> dict:

        # Comparison 1.
        query_to_extract_info = f"""Based on just the two following json objects of resume details and job description details {self.resume_details} and {self.job_description_details},
        check or suggest the following in a list like fashion:
        1. Check if the resume details match the job description requirements and qualifications.
        2. The main differences.
        3. The similarities.
        4. Suggest improvements to the resume.
        """ 
        self.messages.append( {"role": "user", "content": query_to_extract_info} )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        response_text = (response.choices[0].message["content"])

        # Chain in comparison.
        query_specifics = "Provide detailed suggestions that'll make the candidate's chance of getting the said job most probable quoting specific lines in the resume that need to be changed."
        self.messages.append({"role" : "assistant", "content" : response_text})
        self.messages.append({"role" : "user", "content" : query_specifics}) 

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return {"summary": response_text, "specifics": response.choices[0].message["content"] }