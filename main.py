from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver 

template="""
Your an expert in GYM and fitness. You are a helpful assistant that provides accurate and concise answers to questions related to GYM and fitness.
If a general question is asked answer based on your knowledge and if only the user asks about gym available or need then answer from Knowledge base provided by the retriever.
Here is the past conversation, analyse this for better context :{data}
use the memory for better result which stores the previous interactions: {memory}
here is the question:{input}
"""

model=OllamaLLM(model="granite4.1:3b")
prompt=ChatPromptTemplate.from_template(template)
chain= prompt | model

def conversation():
    print("Welcome to GYM FREAKS type 'exit' to end the conversation")
    chat_memory = "" 
    while True:
        user_input = input("\nEnter your queries: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        context = retriver.invoke(user_input)
        res = chain.invoke({
            "memory": chat_memory, 
            "data": context, 
            "input": user_input
        })
        print("\nAI:", res)
        chat_memory += f"User: {user_input}\nAI: {res}\n"
    
if __name__=="__main__":
    conversation()
    