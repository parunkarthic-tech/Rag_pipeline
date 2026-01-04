from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


from google import genai
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))



persistant_directory = "db/chroma_db"

embeeding_model=HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
    
)



db = Chroma(
    persist_directory=persistant_directory,
    embedding_function=embeeding_model,
    collection_metadata ={"hnsw:space":"cosine"}
    
)

query="who is the founder of tesla?"

retriever = db.as_retriever(search_kwargs={"k":3})

relevant_docs=retriever.invoke(query)



context = "\n\n".join(
    [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
)
print(f"(((((((((context)))))))))):  \n{context}")


prompt = f"""
You are a helpful assistant.
Answer the user question using ONLY the context provided.
If the answer is not present in the context, say "I don't know based on the given documents."

Context:
{context}

Question:
{query}

Answer:
"""


for i, doc in enumerate(relevant_docs,1):
    print(f" Document {i} :\n{doc.page_content}\n")
    
    
    
    

print(f"User query: {query}")


for m in client.models.list():
    print(m.name)

response = client.models.generate_content(
    model="models/gemini-flash-latest",
    contents=prompt
)
print("Final Answer:\n")
print(response.text)


    


