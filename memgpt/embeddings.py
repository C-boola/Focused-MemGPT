import typer
import uuid
from typing import Optional, List, Any, Tuple
import os
import numpy as np

from memgpt.utils import is_valid_url, printd
from memgpt.data_types import EmbeddingConfig, Message
from memgpt.credentials import MemGPTCredentials
from memgpt.constants import MAX_EMBEDDING_DIM, EMBEDDING_TO_TOKENIZER_MAP, EMBEDDING_TO_TOKENIZER_DEFAULT, DEFAULT_EMBEDDING_MODEL
from memgpt.config import MemGPTConfig

# from llama_index.core.base.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document as LlamaIndexDocument

# from llama_index.core.base.embeddings import BaseEmbedding
# from llama_index.core.embeddings import BaseEmbedding
# from llama_index.core.base.embeddings.base import BaseEmbedding
# from llama_index.bridge.pydantic import PrivateAttr
# from llama_index.embeddings.base import BaseEmbedding
# from llama_index.embeddings.huggingface_utils import format_text
import tiktoken


def parse_and_chunk_text(text: str, chunk_size: int) -> List[str]:
    parser = SentenceSplitter(chunk_size=chunk_size)
    llama_index_docs = [LlamaIndexDocument(text=text)]
    nodes = parser.get_nodes_from_documents(llama_index_docs)
    return [n.text for n in nodes]


def truncate_text(text: str, max_length: int, encoding) -> str:
    # truncate the text based on max_length and encoding
    encoded_text = encoding.encode(text)[:max_length]
    return encoding.decode(encoded_text)


def check_and_split_text(text: str, embedding_model: str) -> List[str]:
    """Split text into chunks of max_length tokens or less"""

    if embedding_model in EMBEDDING_TO_TOKENIZER_MAP:
        encoding = tiktoken.get_encoding(EMBEDDING_TO_TOKENIZER_MAP[embedding_model])
    else:
        print(f"Warning: couldn't find tokenizer for model {embedding_model}, using default tokenizer {EMBEDDING_TO_TOKENIZER_DEFAULT}")
        encoding = tiktoken.get_encoding(EMBEDDING_TO_TOKENIZER_DEFAULT)

    num_tokens = len(encoding.encode(text))

    # determine max length
    if hasattr(encoding, "max_length"):
        # TODO(fix) this is broken
        max_length = encoding.max_length
    else:
        # TODO: figure out the real number
        printd(f"Warning: couldn't find max_length for tokenizer {embedding_model}, using default max_length 8191")
        max_length = 8191

    # truncate text if too long
    if num_tokens > max_length:
        print(f"Warning: text is too long ({num_tokens} tokens), truncating to {max_length} tokens.")
        # First, apply any necessary formatting
        formatted_text = format_text(text, embedding_model)
        # Then truncate
        text = truncate_text(formatted_text, max_length, encoding)

    return [text]


class EmbeddingEndpoint:
    """Implementation for OpenAI compatible endpoint"""

    # """ Based off llama index https://github.com/run-llama/llama_index/blob/a98bdb8ecee513dc2e880f56674e7fd157d1dc3a/llama_index/embeddings/text_embeddings_inference.py """

    # _user: str = PrivateAttr()
    # _timeout: float = PrivateAttr()
    # _base_url: str = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str,
        user: str,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        if not is_valid_url(base_url):
            raise ValueError(
                f"Embeddings endpoint was provided an invalid URL (set to: '{base_url}'). Make sure embedding_endpoint is set correctly in your MemGPT config."
            )
        self.model_name = model
        self._user = user
        self._base_url = base_url
        self._timeout = timeout

    def _call_api(self, text: str) -> List[float]:
        if not is_valid_url(self._base_url):
            raise ValueError(
                f"Embeddings endpoint does not have a valid URL (set to: '{self._base_url}'). Make sure embedding_endpoint is set correctly in your MemGPT config."
            )
        import httpx

        headers = {"Content-Type": "application/json"}
        json_data = {"input": text, "model": self.model_name, "user": self._user}

        with httpx.Client() as client:
            response = client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json=json_data,
                timeout=self._timeout,
            )

        response_json = response.json()

        if isinstance(response_json, list):
            # embedding directly in response
            embedding = response_json
        elif isinstance(response_json, dict):
            # TEI embedding packaged inside openai-style response
            try:
                embedding = response_json["data"][0]["embedding"]
            except (KeyError, IndexError):
                raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")
        else:
            # unknown response, can't parse
            raise TypeError(f"Got back an unexpected payload from text embedding function, response=\n{response_json}")

        return embedding

    def get_text_embedding(self, text: str) -> List[float]:
        return self._call_api(text)


def default_embedding_model():
    # default to hugging face model running local
    # warning: this is a terrible model
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    model = "BAAI/bge-small-en-v1.5"
    return HuggingFaceEmbedding(model_name=model)


def query_embedding(embedding_model, query_text: str):
    """Generate padded embedding for querying database"""
    query_vec = embedding_model.get_text_embedding(query_text)
    query_vec = np.array(query_vec)
    query_vec = np.pad(query_vec, (0, MAX_EMBEDDING_DIM - query_vec.shape[0]), mode="constant").tolist()
    return query_vec


def embedding_model(config: EmbeddingConfig, user_id: Optional[uuid.UUID] = None):
    """Return LlamaIndex embedding model to use for embeddings"""

    endpoint_type = config.embedding_endpoint_type

    # TODO refactor to pass credentials through args
    credentials = MemGPTCredentials.load()

    if endpoint_type == "openai":
        assert credentials.openai_key is not None
        from llama_index.embeddings.openai import OpenAIEmbedding

        additional_kwargs = {"user_id": user_id} if user_id else {}
        model = OpenAIEmbedding(
            api_base=config.embedding_endpoint,
            api_key=credentials.openai_key,
            additional_kwargs=additional_kwargs,
        )
        return model

    elif endpoint_type == "azure":
        assert all(
            [
                credentials.azure_key is not None,
                credentials.azure_embedding_endpoint is not None,
                credentials.azure_version is not None,
            ]
        )
        from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
        model = "text-embedding-ada-002"
        deployment = credentials.azure_embedding_deployment if credentials.azure_embedding_deployment is not None else model
        return AzureOpenAIEmbedding(
            model=model,
            deployment_name=deployment,
            api_key=credentials.azure_key,
            azure_endpoint=credentials.azure_endpoint,
            api_version=credentials.azure_version,
        )

    elif endpoint_type == "hugging-face":
        return EmbeddingEndpoint(
            model=config.embedding_model,
            base_url=config.embedding_endpoint,
            user=user_id,
        )

    else:
        return default_embedding_model()


def create_embedding(text: str, embedding_config: EmbeddingConfig, model: Optional[str] = None) -> List[float]:
    """
    Generates an embedding for the given text using the specified configuration.
    This function acts as a wrapper around the LlamaIndex embedding model logic.
    """
    # Initialize the embedding model using the agent's config
    # The 'model' parameter here is for potentially overriding the model from embedding_config if needed,
    # but embedding_model function primarily uses embedding_config.
    
    # Extract user_id from embedding_config if available and relevant for the model
    # user_id_to_pass = None # Placeholder, adjust if user_id needs to be sourced differently or passed through
    # For instance, if embedding_config could carry a user_id or if it's globally available
    
    embed_model = embedding_model(config=embedding_config) # user_id can be passed if available in config/context

    # Check if the text needs to be split due to length constraints
    # The specific model name might be in embedding_config.embedding_model
    texts_to_embed = check_and_split_text(text, embedding_config.embedding_model if embedding_config else DEFAULT_EMBEDDING_MODEL)
    
    # For simplicity, if splitting occurs, we might average embeddings or handle appropriately.
    # Here, assuming check_and_split_text returns a list and we use the first part,
    # or that it handles overly long text by truncation/error.
    # A more robust implementation might embed chunks and average them.
    if not texts_to_embed:
        raise ValueError("Text resulted in no content after splitting/checking.")

    # LlamaIndex's BaseEmbedding models usually have a get_text_embedding or similar method.
    # If it's a list of texts (e.g. from chunking), it might be get_text_embedding_batch.
    # We'll assume get_text_embedding for a single string for now.
    embedding_vector = embed_model.get_text_embedding(texts_to_embed[0])
    
    # Ensure the embedding is a list of floats and pad if necessary (though padding is often context-specific)
    if not isinstance(embedding_vector, list) or not all(isinstance(x, float) for x in embedding_vector):
        # Try to convert if it's a numpy array, common for some embedding libraries
        if hasattr(embedding_vector, 'tolist'):
            embedding_vector = embedding_vector.tolist()
        else:
            raise TypeError(f"Embedding generation did not return a list of floats. Got: {type(embedding_vector)}")

    # Padding to MAX_EMBEDDING_DIM is usually for storage in a DB with fixed vector sizes.
    # If create_embedding is for general use, this might be optional or handled by the caller.
    # query_embedding function already handles padding.
    # For now, returning raw embedding from the model.
    return embedding_vector


def create_message_pair_embeddings(
    message_sequence: List[Message],
    embedding_config: EmbeddingConfig,
) -> List[Tuple[uuid.UUID, uuid.UUID, List[float]]]:
    """
    Creates vector embeddings for message pairs (user message and subsequent assistant message).

    Args:
        message_sequence: A list of Message objects.
        embedding_config: The embedding configuration for the agent.

    Returns:
        A list of tuples, where each tuple contains:
        (user_message_id, assistant_message_id, combined_embedding_vector).
    """
    pair_embeddings = []
    if not message_sequence or len(message_sequence) < 2:
        return pair_embeddings

    # Find the create_embedding function, it might be local or needs to be dynamically accessed
    # For this edit, we assume create_embedding is available in the scope.
    # If create_embedding is a global function in this file, this direct call is fine.

    for i in range(len(message_sequence) - 1):
        msg1 = message_sequence[i]
        msg2 = message_sequence[i+1]

        if msg1.role == "user" and msg2.role == "assistant":
            # Ensure text is not None and IDs are not None
            text1 = msg1.text if msg1.text is not None else ""
            text2 = msg2.text if msg2.text is not None else ""
            
            if msg1.id is None or msg2.id is None:
                print(f"Warning: Skipping message pair due to missing ID(s). User msg ID: {msg1.id}, Assistant msg ID: {msg2.id}")
                continue

            combined_text = text1.strip() + " " + text2.strip() # Simple concatenation with a space

            # Skip embedding if combined text is empty or only whitespace
            if not combined_text.strip():
                continue

            try:
                # This relies on 'create_embedding' being defined and accessible in this file's scope
                embedding_vector = create_embedding(
                    text=combined_text,
                    embedding_config=embedding_config,
                    # The 'create_embedding' function should handle which specific model to use
                    # based on embedding_config or its own logic.
                )
                pair_embeddings.append((msg1.id, msg2.id, embedding_vector))
            except Exception as e:
                print(f"Error creating embedding for pair (user_msg_id={msg1.id}, assistant_msg_id={msg2.id}): {e}")
                # Optionally skip this pair or append a placeholder
                continue

    return pair_embeddings


def calculate_centroid(embedding_vectors: List[List[float]]) -> Optional[np.ndarray]:
    """
    Calculates the centroid (mean vector) of a list of embedding vectors.

    Args:
        embedding_vectors: A list of embedding vectors (list of lists of floats).

    Returns:
        A numpy array representing the centroid, or None if the input is empty.
    """
    if not embedding_vectors:
        return None
    
    # Convert list of lists to a 2D numpy array
    vectors_array = np.array(embedding_vectors)
    
    # Calculate the mean across the 0-th axis (column-wise mean for each dimension)
    centroid = np.mean(vectors_array, axis=0)
    return centroid


def calculate_cosine_distances(target_vector: np.ndarray, embedding_vectors: List[np.ndarray]) -> List[float]:
    """
    Calculates the cosine distance of each vector in a list from a target vector.

    Args:
        target_vector: The vector from which distances are measured (e.g., a centroid).
        embedding_vectors: A list of numpy arrays (embedding vectors).

    Returns:
        A list of cosine distances.
    """
    distances = []
    if target_vector is None or not embedding_vectors:
        return distances

    # Normalize the target vector
    norm_target = np.linalg.norm(target_vector)
    if norm_target == 0: # Avoid division by zero for a zero vector
        # If target is zero vector, distance is undefined or 1 (if vectors are non-zero)
        # or 0 (if vector is also zero). Let's assume 1 for non-zero vectors.
        # For simplicity, if centroid is zero, all distances are effectively max (or handle as error)
        return [1.0] * len(embedding_vectors) 
    
    normalized_target_vector = target_vector / norm_target

    for vec in embedding_vectors:
        norm_vec = np.linalg.norm(vec)
        if norm_vec == 0: # Avoid division by zero for a zero vector
            distances.append(1.0) # Distance to a zero vector is 1 (max)
            continue
        
        normalized_vec = vec / norm_vec
        
        # Cosine similarity
        similarity = np.dot(normalized_target_vector, normalized_vec)
        
        # Cosine distance
        distance = 1 - similarity
        distances.append(distance)
        
    return distances
