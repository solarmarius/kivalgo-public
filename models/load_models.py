"""
Contains some functions to initialize models that are chosen.
To standarize and provide easy editing of model
"""

import sys
import os

# To import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.constants import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
)


def init_tokenizer():
    # Check to see if model is installed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    llm_model_path = os.path.join(current_dir, LLM_MODEL_NAME)
    if not os.path.exists(llm_model_path):
        raise FileNotFoundError(
            f"The model {LLM_MODEL_NAME} was not found. Have you installed the model using git lfs and git clone?"
        )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, local_files_only=True)
    return tokenizer


def init_model():
    # Check to see if model is installed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    llm_model_path = os.path.join(current_dir, LLM_MODEL_NAME)
    if not os.path.exists(llm_model_path):
        raise FileNotFoundError(
            f"The model {LLM_MODEL_NAME} was not found. Have you installed the model using git lfs and git clone?"
        )

    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, local_files_only=True)
    return model


def init_embedding_model():
    # Check to see if model is installed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    llm_model_path = os.path.join(current_dir, EMBEDDING_MODEL_NAME)
    if not os.path.exists(llm_model_path):
        raise FileNotFoundError(
            f"The embedding model {EMBEDDING_MODEL_NAME} was not found. Have you installed the model using git lfs and git clone?"
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
    )
    return embedding_model
