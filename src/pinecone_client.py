"""
Pinecone vector database client.
Handles index creation, upserting, and querying.
"""

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 1536))

_pc = None
_index = None


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def init_index(
    name: str = INDEX_NAME,
    dims: int = EMBEDDING_DIMENSIONS,
    metric: str = "cosine",
) -> None:
    """Create the Pinecone index if it does not already exist."""
    global _index
    pc = _get_client()
    existing = [idx.name for idx in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dims,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index '{name}' ({dims}d, {metric})")
    else:
        print(f"Using existing Pinecone index '{name}'")
    _index = pc.Index(name)


def _get_index():
    global _index
    if _index is None:
        init_index()
    return _index


def upsert_vector(
    vector_id: str,
    vector: list[float],
    metadata: dict,
    namespace: str = "",
) -> None:
    """Upsert a single embedding vector with metadata."""
    idx = _get_index()
    idx.upsert(
        vectors=[{"id": vector_id, "values": vector, "metadata": metadata}],
        namespace=namespace,
    )


def upsert_batch(
    records: list[dict],
    namespace: str = "",
    batch_size: int = 100,
) -> None:
    """
    Upsert a batch of records.
    Each record: {"id": str, "values": list[float], "metadata": dict}
    """
    idx = _get_index()
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        idx.upsert(vectors=batch, namespace=namespace)
    print(f"Upserted {len(records)} vectors to '{INDEX_NAME}'")


def query_vectors(
    vector: list[float],
    top_k: int = 5,
    filter: dict = None,
    namespace: str = "",
    include_metadata: bool = True,
) -> list[dict]:
    """
    Query the index for the top_k most similar vectors.
    Returns a list of matches with id, score, and metadata.
    """
    idx = _get_index()
    kwargs = {
        "vector": vector,
        "top_k": top_k,
        "namespace": namespace,
        "include_metadata": include_metadata,
    }
    if filter:
        kwargs["filter"] = filter

    response = idx.query(**kwargs)
    return [
        {
            "id": m["id"],
            "score": m["score"],
            "metadata": m.get("metadata", {}),
        }
        for m in response["matches"]
    ]


def delete_vector(vector_id: str, namespace: str = "") -> None:
    """Delete a single vector by ID."""
    _get_index().delete(ids=[vector_id], namespace=namespace)


def describe_index() -> dict:
    """Return index stats (vector count, dimensions, etc.)."""
    return _get_index().describe_index_stats()
