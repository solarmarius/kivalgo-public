import sys
import os

# To import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import DATA_PATH
from loaders.chroma_database import init_wiki_database, init_textsplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

if __name__ == "__main__":
    parti_path = DATA_PATH + "/wiki"
    loader = DirectoryLoader(parti_path, loader_cls=TextLoader, show_progress=True)
    documents = loader.load()
    text_splitter = init_textsplitter()
    texts = text_splitter.split_documents(documents)

    wiki_database = init_wiki_database()
    wiki_database.add_documents(documents=texts)
