"""
Text ingestion: reads .txt / .md files, chunks them,
embeds each chunk with Gemini Embedding 2, and upserts to Pinecone.
"""

import os
import uuid
from pathlib import Path
from tqdm import tqdm

from src.embedder import embed_text
from src.pinecone_client import upsert_batch, init_index

# ~500 tokens ≈ 2000 characters (rough approximation)
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def ingest_text_file(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    namespace: str = "",
) -> int:
    """
    Ingest a text/markdown file into Pinecone.
    Returns the number of chunks upserted.
    """
    init_index()
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    chunks = _chunk_text(text, chunk_size, overlap)

    records = []
    for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {path.name}")):
        vector = embed_text(chunk, task_type="RETRIEVAL_DOCUMENT")
        records.append(
            {
                "id": f"{path.stem}-{i}-{uuid.uuid4().hex[:8]}",
                "values": vector,
                "metadata": {
                    "source": str(path),
                    "modality": "text",
                    "chunk_index": i,
                    "content_preview": chunk[:200],
                },
            }
        )

    upsert_batch(records, namespace=namespace)
    print(f"Ingested {len(records)} text chunks from '{path.name}'")
    return len(records)
