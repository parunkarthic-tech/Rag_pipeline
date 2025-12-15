import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def main():
    check_env=os.getenv("CHECK_KEY")
    print(f"In the main: {check_env}")


if __name__ == "__main__":
    main() 




