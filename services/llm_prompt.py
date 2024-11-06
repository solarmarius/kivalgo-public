import streamlit as st
import sys
import os

# To import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import LLM_MODEL_NAME

from models.load_models import init_tokenizer, init_model
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_openai.chat_models import ChatOpenAI


def query_llm(query):
    print("Querying the model")
    model = HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL_NAME,
        task="text-generation",
        model_kwargs={
            "trust_remote_code": True,
        },
        pipeline_kwargs={
            "max_new_tokens": 500,
            "return_full_text": False,
            "do_sample": True,
        },
    )

    print("Creating the output")
    output = model.invoke(query)

    print(output)
    return output.content
