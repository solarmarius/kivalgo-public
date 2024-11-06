import sys
import os

# To import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import DATA_PATH
from loaders.chroma_database import init_parti_database, init_textsplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

if __name__ == "__main__":
    parti_path = DATA_PATH + "/parti"
    loader = DirectoryLoader(parti_path, loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    text_splitter = init_textsplitter()
    texts = text_splitter.split_documents(documents)

    parti_database = init_parti_database()
    parti_database.add_documents(documents=texts)
