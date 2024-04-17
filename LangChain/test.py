from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader('../data', glob="**/*.txt", show_progress=True)
docs = loader.load()

print(len(docs)