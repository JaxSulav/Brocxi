import os
from dotenv import load_dotenv
from transformers import AutoModel
from langchain.storage import LocalFileStore
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



# cache_store = LocalFileStore("./mxbai_cache_v2/")

# Load txt files from dir
loader = DirectoryLoader('../extracted_files', glob="*.txt", loader_cls=TextLoader, show_progress=True)
docs = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=256,
    chunk_overlap=64,
)
chunked = text_splitter.split_documents(docs)

# model = AutoModel.from_pretrained('mixedbread-ai/mxbai-embed-large-v1', trust_remote_code=True) 

model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {'device': 'cpu'}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

# embeddings_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings_model, cache_store, namespace="mixedbread-ai/mxbai-embed-large-v1")

db = FAISS.from_documents(chunked, cached_embedder)

db.save_local("mxbai_faiss_index_v2")

print("Embeddings saved ...")
