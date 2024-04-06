from typing import List
import time
from nomic import embed

def generate_embeddings(input_texts: List[str], model_api_string: str, task_type="search_document") -> List[List[float]]:
    """Generate embeddings from Nomic Embedding API.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.
        task_type: str. the task type for the embedding model. Defaults to "search_document". One of `search_query`, `search_document`, `classification`, or `clustering`.

    Returns:
        a list of embeddings. Each element corresponds to the each input text.
    """
    start = time.time()
    outputs = embed.text(
        texts=[f"{task_type}: {text}" for text in input_texts],  # prefix is used to identify the input text.
        model=model_api_string,
        task_type=task_type,
    )

    print(f"Embedding generation took {str(time.time() - start)} seconds.")
    return outputs["embeddings"]

embedding_model_string = 'nomic-embed-text-v1' # only one model is available at the moment.
vector_database_field_name = 'nomic-embed-text'  # define your embedding field name.
NUM_DOC_LIMIT = 250  # the number of documents you will process and generate embeddings.

sample_output = generate_embeddings(["This is a test."], embedding_model_string)
print(f"Embedding size is: {str(len(sample_output[0]))}")