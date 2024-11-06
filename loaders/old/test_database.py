"""
To test the database
"""

import sys
import os

# To import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.constants import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PATH,
    PROMPT_TEMPLATE,
    LLM_MODEL_NAME,
)
from models.load_models import init_tokenizer, init_model, init_embedding_model

from loaders.chroma_database import init_database

DEVICE = "mps"


def query_rag(query_text):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    Args:
      - query_text (str): The text to query the RAG system with.
    Returns:
      - formatted_response (str): Formatted response including the generated text and sources.
      - response_text (str): The generated response text.
    """
    db = init_database()

    print("Finding most similar documents to query...")

    # Retrieving the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("Prompt to the model is:", prompt)

    tokenizer = init_tokenizer()
    model = init_model()

    input = tokenizer(prompt, return_tensors="pt")
    input_ids = input.input_ids
    attention_mask = input.attention_mask

    print("Generating response...")

    # Generate response text based on the prompt
    generated_ids = model.generate(
        input_ids,
        max_length=2000,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("Decoding response...")

    response_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    print("The response is:")
    print(response_text)

    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]

    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text


if __name__ == "__main__":
    query_text = "Når ble de ulike partiene grunnlagt?"

    # Let's call our function we have defined
    formatted_response, response_text = query_rag(query_text)
    # and finally, inspect our final response!
    print(response_text)
