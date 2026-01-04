from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


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

query="who is elon musk?"

retriever = db.as_retriever(search_kwargs={"k":3})

relevant_docs=retriever.invoke(query)




print(f"User query: {query}")

for i, doc in enumerate(relevant_docs,1):
    print(f" Document {i} :\n{doc.page_content}\n")


