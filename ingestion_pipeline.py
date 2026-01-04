import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_chroma import Chroma
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def main():
    check_env=os.getenv("CHECK_KEY")
    print(f"In the main: {check_env}")


#1.Loading
def load_documents(doc_path="docs"):
    "load all the documents from the directory"
    print(f"(❁´◡`❁) Loading documents from {doc_path} ...... ")
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"The directory : {doc_path} doesn't exists create and  add the text files ")
    
    loader = DirectoryLoader(
        path=doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    documents = loader.load()
    
    if(len(documents)==0):
        raise FileNotFoundError(f"No .txt find in this patg: {doc_path}. Add the .txt files.")
    
    for i, doc in enumerate(documents[0:2]):
        print(f"\n Documents count: {i+1}")
        print(f"\n Source: {doc.metadata['source']}")
        print(f"\n content length : {len(doc.page_content)}")
        print(f"\n doc content: {doc.page_content[0:100]} ........")
        print(f"\n metadata: {doc.metadata}")
        
    return documents

#2)Splitting

def split_doc(documents,chunk_size=1000,chunk_overlap=0):
    "split the documents into chunks"
    print("(●'◡'●) Splitting documents")
    
    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks=text_splitter.split_documents(documents)
    
    if chunks:
        
        print("chunk got chunked ")
        
        
    for i, chunk in enumerate(chunks[0:5]):
        print(f"\n chunk count {i+1}")
        print(f"\n chunk source { chunk.metadata["source"]}")
        print(f"\n length: {len(chunk.page_content)} character")
        print(f"\n page content: {chunk.page_content}")
        
    
    if len(chunks)>5:
        print(f"there are more chunks: {len(chunks)-5}")
        
    return chunks

#3) Storing

def create_store_vector(chunks,persist_directory="db/chroma_db"):
    "creats and store the embedding in chrom db"
    
    print("created embedding and storing it in chroma db in {persist_directory}")
    
    embedding_model=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vector_store= Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    
    print("vector store created.....")
    
    return vector_store
    
    

if __name__ == "__main__":
    main() 
    documents=load_documents()
    chunks=split_doc(documents)
    vector_db=create_store_vector(chunks)




