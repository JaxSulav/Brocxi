
import os
import torch
import transformers
import chainlit as cl
from getpass import getpass
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModel
from langchain import HuggingFaceHub
from langchain_community.llms import Ollama
from langchain.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# HUGGINGFACEHUB_API_TOKEN = getpass()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
load_dotenv()

# HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
# login(token = HUGGINGFACE_TOKEN)


# embeddings_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# from transformers import AutoModel

embeddings_model = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    model_kwargs={'device': 'cpu'},
)

db = FAISS.load_local("mxbai_faiss_index_v2", embeddings_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# model_id = "LoneStriker/opus-v1.2-llama-3-8b-GGUF"

# llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_TOKEN, repo_id=model_id, model_kwargs={
#     "temperature": 0.5, "max_new_tokens": 150
# })


llm = Ollama(model="llama3", temperature=0)

@cl.on_chat_start
async def on_chat_start():

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=compression_retriever,
        memory=memory,
        return_source_documents=True,
    )

    cl.user_session.set("chain", chain)

#TODO: Stream response
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  

    text_elements = [] 

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements, author="Brocxi").send()
