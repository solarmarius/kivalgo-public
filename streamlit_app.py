import streamlit as st
import time
import numpy as np
import pandas as pd
import sys
import os

# To import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.k_nearest import find_k_documents
from utils.create_context import create_context
from services.llm_prompt import query_llm


def generate_response(prompt, n_docs):
    st.write_stream(
        write_stream(
            "Jeg prøver å finne relevante dokumenter for å svare på ditt spørsmål 📚"
        )
    )
    documents = find_k_documents(prompt, n_docs)
    final_prompt = create_context(prompt, documents)
    st.write_stream(
        write_stream("Nå spørr jeg min store språkmodell for svar! Vennligst vent ⏳")
    )
    response = query_llm(final_prompt)
    st.write_stream(write_stream(response))


def write_stream(response):
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Streamlit app layout
st.title("KIVALGO")
st.caption(
    "Velkommen til fremtiden i demokrati. Skriv under dine spørsmål i fortroelighet. Vi lagrer IKKE din data!"
)

n_docs = st.radio(
    "Antall partier",
    [2, 9],
    captions=["En/noen få partier", "Alle partiene"],
    horizontal=True,
    help="Hvor mange partier vil du vite noe om? En/få partier gir mer utdypende svar.",
)

st.divider()

# st.button("Velg spørretingtang")
# 2
# alle partier -> 9

# with st.chat_message("assistant"):
#    st.write("KIVA: Hei👋 Mitt navn er KIVA! Hva kan jeg hjelpe deg med i dag?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

if prompt == None:
    print("Gjør ingenting!")
    response = "KIVA: Hei👋 Mitt navn er KIVA! Hva kan jeg hjelpe deg med i dag?"
    with st.chat_message("assistant"):
        st.write_stream(write_stream(response))
else:
    print("INPUT prompt:", prompt)
    with st.chat_message("assistant"):
        Response = generate_response(prompt, n_docs=n_docs)

# Add assistant response to chat history
# st.session_state.messages.append({"role": "assistant", "content": Response})
