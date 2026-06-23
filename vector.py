from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
import os

DB_DIR = "./chroma_langchain_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")

if os.path.exists(DB_DIR):
    print("Loading existing vector database...")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="gym_rag"
    )
else:
    print("Database not found. Creating and embedding documents now (this may take a minute)...")
    loader = CSVLoader("gym_rag.csv")
    documents = loader.load()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="gym_rag"
    )

retriver = vectorstore.as_retriever(search_kwargs={"k": 5})