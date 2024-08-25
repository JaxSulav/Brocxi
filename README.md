# Brocxi
<hr>

## Brocxi is a gaming assistant/guide who specializes in certain specific games and will guide you through different phases of your gameplay. You can ask Brocxi about defeating bosses, finding runes, side quests, and so on.

#### This repository tests different embedding models, vector databases, RAG pipelines, and Frameworks, Evaluations, and implements the open source models for RAG setup.


### To run Brocxi locally:

> Setup your Python environment using requirements

`$ pip install virtualenv`<br>
`$ virtualenv env -p=python3.11`<br>
`$ pip install -r requirements.txt`

> Activate your virtual environment:<br>
`$ source env/bin/activate`

> Install Ollama if you want to use Ollama:<br>
 `https://ollama.com/`

If you want to use LlamaCpp, Langchain's LlamaCpp is used.

If you want to use APIs to perform embeddings or run HuggingFace models. You need to set your key in the .env file inside the app dir.

> Reference LangChain/inject.py if you want to vectorize and inject embeddings into vector db. Here, the saved cache is already uploaded so you do not need to do that.

> Launch ChainLit App:<br>
`$ cd LangChain/app`<br>
`$ chainlit run app.py`



