"""
Databases object init script,
use this file to start the databases for quering. 
"""

import sys
import os

# To import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.constants import (
    CHROMA_PATH,
    CHROMA_BASE_COLLECTION,
    CHROMA_PARTI_COLLECTION,
    CHROMA_WIKI_COLLECTION,
)

from models.load_models import init_embedding_model


def init_base_database():
    embedding_function = init_embedding_model()
    chroma_client = Chroma(
        collection_name=CHROMA_BASE_COLLECTION,
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH,
    )
    return chroma_client


def init_wiki_database():
    embedding_function = init_embedding_model()
    chroma_client = Chroma(
        collection_name=CHROMA_WIKI_COLLECTION,
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH,
    )
    return chroma_client


def init_parti_database():
    embedding_function = init_embedding_model()
    chroma_client = Chroma(
        collection_name=CHROMA_PARTI_COLLECTION,
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH,
    )
    return chroma_client


def init_textsplitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=int(512 / 10),
        add_start_index=True,
        strip_whitespace=True,
    )
    return text_splitter


if __name__ == "__main__":
    # base_database = init_base_database()
    wiki_database = init_wiki_database()
    # parti_database = init_parti_database()
    result = wiki_database.similarity_search(query="Hva mener partiene om miljø?")
    for doc in result:
        print(doc)
