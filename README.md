# Resume Chain and Cover Letter Generator

This repository contains the code for 2 deployed apps:

1. __Resume Chain__: This app takes in a Resume and a Job Description and highlights similarities, differences and makes recommendations. Website: https://resumecomparer.streamlit.app/
2. __Cover Letter Generator__: This app takes the contents of a Resume and converts it into a cover letter. Website: https://coverletter-generator.streamlit.app/

Both the applications were written by the help of langchain and OpenAI's LLMs and run and hosted with streamlit.

## Installation

1. Ensure you have python 3.8+.
2. Clone the repo: `git clone https://github.com/MokoSan/ResumeChain.git`.
3. Optionally create a virtualenv ([here](https://realpython.com/python-virtual-environments-a-primer/) is a good resource to get started) and run: ``pip install -r requirements.txt``.
   1. Ensure you have `streamlit` in your path.
4. Once the requirements are installed, run the applications:
   1. __Resume Chain__ 
      1. `cd apps/resume_chain_app`.
      2. Copy the .env.example file into a .env file and enter your OpenAI API KEY for the OPENAI_API_KEY environment variable.
         1. Instructions on how to obtain this can be found [here](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/).
      3. `streamlit run .\streamlitui.py`.
   2. __Cover Letter Generator__
      1. `cd apps/coverletter_generator_app`.
      2. Copy the .env.example file into a .env file and enter your OpenAI API KEY for the OPENAI_API_KEY environment variable.
         1. Instructions on how to obtain this can be found [here](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/).
      3. `streamlit run .\streamlitui.py`.

## Contributions

Contributions are welcomed and encouraged! If you run into any problems, create issues on this repository.

## License

This project is licensed under the [MIT License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).

## TODOs

1. Add a destructor to give memory back after the comparison is complete from the in-memory vector databases. 
2. Improve the OpenAI Chat Completion calls.
3. Add more documentation.