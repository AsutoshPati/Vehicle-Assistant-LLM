import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Path to your local model (download and save)
MODEL_DIR = os.environ.get("MODEL_PATH", "model")
MODEL_NAME = os.environ.get(
    "TRANSFORMER_MODEL", "sentence-transformers/all-mpnet-base-v2"
)

transformer_name, only_model_name = MODEL_NAME.split("/")
model_loc = os.path.join(MODEL_DIR, transformer_name, only_model_name)
os.makedirs(model_loc, exist_ok=True)


def get_embedding_model():
    global MODEL_NAME, model_loc

    def is_directory_empty(directory_path):
        return not any(os.scandir(directory_path))

    if is_directory_empty(model_loc):
        print(f"** Downloading model {MODEL_NAME}")

        # You may need to log in for downloading the model
        from huggingface_hub import login

        login()

        # Load the model from Hugging Face Hub
        model = SentenceTransformer(MODEL_NAME)

        # Save the model locally
        model.save(model_loc)

        print(f"Model saved to {model_loc}")

    embeddings = HuggingFaceEmbeddings(model_name=model_loc)
    return embeddings


if __name__ == "__main__":
    embedding = get_embedding_model()

    # Use the embedding instance (Sample)
    documents = ["hello world"]
    output = embedding.embed_documents(documents)
    print(output)
