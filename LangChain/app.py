import os
import chainlit as cl
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.contextual_compression import \
    ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

COHERE_API_KEY = os.getenv('COHERE_API_KEY')

load_dotenv()

# embeddings_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel

model = AutoModel.from_pretrained('mixedbread-ai/mxbai-embed-large-v1', trust_remote_code=True) 

model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {'device': 'cpu'}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

db = FAISS.load_local("mxbai_faiss_index", embeddings_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
llm = Ollama(model="mistral", temperature=0)



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

    await cl.Message(content=answer, elements=text_elements).send()
