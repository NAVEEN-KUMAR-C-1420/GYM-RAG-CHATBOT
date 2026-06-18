from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df=pd.read_csv("gym_rag.csv")

Embeddings=OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")

db_loc="./chroma_langchain_db"
add_doc=not os.path.exists(db_loc)

if add_doc:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=f"""
            Gym Name: {row["gym_name"]}
            Location: {row["location"]}
            Membership: {row["membership_type"]}
            Monthly Fee: {row["monthly_fee_inr"]}
            Personal Training Fee: {row["personal_training_fee_inr"]}
            Diet Plan: {row["diet_plan"]}
            Workout: {row["recommended_workout"]}
            Trainer Level: {row["trainer_level"]}
            Rating: {row["rating"]}
            """,
            metadata={
                "gym_id": row["gym_id"],
                "location": row["location"],
                "diet_plan": row["diet_plan"],
                "workout": row["recommended_workout"],
                "rating": row["rating"]
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document) 

vectorstore=Chroma(
    collection_name="gym_collection",
    embedding_function=Embeddings,
    persist_directory=db_loc
)

if add_doc:
    vectorstore.add_documents(documents, ids=ids)

retriver=vectorstore.as_retriever(search_kwargs={"k":5})
