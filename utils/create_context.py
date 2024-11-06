from utils.constants import PROMPT_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate


def create_context(user_prompt: str, documents: tuple):
    docs = documents[0]
    context_text = "\n\n - -\n\n".join(
        [doc.page_content.replace("\n", "") for doc in docs]
    )
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    full_prompt = prompt_template.format(user_input=user_prompt, context=context_text)
    return full_prompt
