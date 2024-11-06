"""
This file finds and returns the nearest neighbour documents in the databases given a query.
The three databases are prioritized in the following order: parti - wiki - base
"""

import sys
import os

# To import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.chroma_database import (
    init_base_database,
    init_wiki_database,
    init_parti_database,
)

from utils.constants import PARTI_SCORE, BASE_SCORE, WIKI_SCORE


def find_k_documents(query, docs, k=10) -> tuple:
    # k: how many documents are returned from each database if they are similar
    # The function returns a list of all similar documents

    # Collect results
    all_results = []
    documents_metadata = []

    # Search in parti database
    parti_db = init_parti_database()
    results_parti = parti_db.similarity_search_with_relevance_scores(query, k=k)
    if results_parti:
        # Only add documents with a score over 0.4
        all_results.extend([doc for doc, score in results_parti if score > PARTI_SCORE])

    if len(all_results) > docs:
        for doc in all_results:
            documents_metadata.append(doc.metadata)
        return (all_results, documents_metadata)

    # Search in wiki database
    wiki_db = init_wiki_database()
    results_wiki = wiki_db.similarity_search_with_relevance_scores(query, k=k)
    if results_wiki:
        # Only add documents with a score over 0.6
        all_results.extend([doc for doc, score in results_wiki if score > WIKI_SCORE])

    if len(all_results) > docs:
        for doc in all_results:
            documents_metadata.append(doc.metadata)
        return (all_results, documents_metadata)

    # Search in base database
    base_db = init_base_database()
    results_base = base_db.similarity_search_with_relevance_scores(query, k=k)
    if results_base:
        # Only add documents with a score over 0.8
        all_results.extend([doc for doc, score in results_base if score > BASE_SCORE])

    for doc in all_results:
        documents_metadata.append(doc.metadata)
    return (all_results, documents_metadata)


if __name__ == "__main__":
    docs = find_k_documents("Hva mener partiene om miljø?", 2)
    documents_metadata = []
    print(docs)
    for doc in docs:
        documents_metadata.append(doc.metadata)
        print(doc)
        print(type(doc))
    print(type(docs))

    context_text = "\n\n - -\n\n".join(
        [doc.page_content.replace("\n", "") for doc in docs]
    )
    print(context_text)

    print(documents_metadata)
