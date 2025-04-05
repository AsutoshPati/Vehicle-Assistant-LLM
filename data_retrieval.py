import gc
import os
import time

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma

from download_model import get_embedding_model

load_dotenv()
CHROMA_PATH = os.environ.get("CHROMA_PATH", "knowledge_library")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama3.2:1b")

PROMPT_TEMPLATE = """
Answer the question in maximum 50 words based only on the following context:

{context}

---

Answer the question in maximum 50 words based on the above context: {question}
"""


def query_rag(query_text: str):
    start_time = time.time()

    top_n = 5
    embedding_model = get_embedding_model()

    vectordb = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_model
    )

    # Search the DB.
    results = vectordb.similarity_search_with_score(query_text, k=top_n)
    # print("Search Result: ", results)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results]
    )
    # print("Context: ", context_text)

    # Create prompt with context & query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Send the prompt with context and query
    print(f"Sending prompt to ({LLAMA_MODEL})")
    model = Ollama(model=LLAMA_MODEL)
    response_text = model.invoke(prompt)
    # print("Response:", response_text)

    # Add source to the response text
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = {"AI Response": response_text, "Sources": sources}
    print("Formatted Response:", formatted_response)

    print(f"AI response took {time.time()-start_time} seconds")

    return formatted_response


if __name__ == "__main__":

    query = "Give tribute to Mr. Ratan Tata"
    query_rag(query)
