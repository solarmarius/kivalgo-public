"""
This script creates a vector databases (Chroma) from 
the files contained in /data.

NOTE: running this file with replace the existing database!
And it takes a long time to create a database. 
"""

import sys
import os
import shutil

# To import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from typing import Optional, List
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma

from utils.constants import EMBEDDING_MODEL_NAME, DATA_PATH
from loaders.chroma_database import (
    init_base_database,
    init_wiki_database,
    init_parti_database,
)


def load_pdf_documents():
    parti_path = DATA_PATH + "/parti"
    document_loader = PyPDFDirectoryLoader(parti_path)
    return document_loader.load()


def load(file_name):
    f = open(file_name, "r")
    text = f.read()
    f.close()
    return LangchainDocument(page_content=text, metadata={"name": file_name})


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str],
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    # TODO get ids
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def save_to_chroma(chunks: list[Document], database: Chroma):
    """
    Save the given list of Document objects to a Chroma database.
    Args:
    chunks (list[Document]): List of Document objects representing text chunks to save.
    Returns:
    None
    """
    database.add_documents(documents=chunks)
    print(f"Saved {len(chunks)} chunks to {database}.")


if __name__ == "__main__":
    # disable warning from HF
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Parti database
    print("Processing parti files")
    data_parti_files = load_pdf_documents()

    print("Chunking documents...")
    docs_processed = split_documents(
        512,  # We choose a chunk size adapted to our model
        data_parti_files,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    parti_database = init_wiki_database()

    print("Saving parti documents to database...")
    save_to_chroma(docs_processed, parti_database)

    # Base database
    # data_parti_files = [f"../data/{f}" for f in os.listdir(DATA_PATH)]
    # raw_parti_knowledge_base = [load(file) for file in data_parti_files]
