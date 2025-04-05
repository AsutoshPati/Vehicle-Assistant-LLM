import os
import shutil

from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_chroma import Chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from download_model import get_embedding_model

load_dotenv()

# Path of knowledge for context/training (Only pdf files)
DATA_PATH = os.environ.get("DATA_PATH", "knowledge_base")
os.makedirs(DATA_PATH, exist_ok=True)

# DB to store knowledge vector embeddings
CHROMA_PATH = os.environ.get("CHROMA_PATH", "knowledge_library")


def clear_database() -> None:
    # CAUTION - This will delete the existing knowledgebase
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    print("DB cleanup completed")


def load_documents(data_dir: str) -> list:
    # Load pdf documents from a directory
    print(f"Reading documents from {data_dir}")
    document_loader = PyPDFDirectoryLoader(data_dir)
    return document_loader.load()


def split_documents(document_list: list[Document]):
    # Split document size by chunk size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(document_list)


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def add_knowledge_to_chroma(chunks):
    embedding_model = get_embedding_model()

    # Load the existing database.
    vectordb = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_model
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = vectordb.get(
        include=[]
    )  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vectordb = Chroma.from_documents(
            documents=new_chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_PATH,
            ids=new_chunk_ids,
        )
        vectordb.persist()
    else:
        print("✅ No new documents to add")


if __name__ == "__main__":
    # Uncomment the following code line to clean the DB data completely
    # clear_database()

    documents = load_documents(DATA_PATH)
    print(documents)

    doc_chunks = split_documents(documents)
    print(doc_chunks)

    add_knowledge_to_chroma(doc_chunks)
