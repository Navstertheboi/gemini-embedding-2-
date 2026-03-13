"""
Image ingestion: embeds PNG/JPEG images with Gemini Embedding 2
and upserts to Pinecone.
"""

import uuid
from pathlib import Path

from src.embedder import embed_image
from src.pinecone_client import upsert_vector, init_index

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ingest_image_file(file_path: str, namespace: str = "") -> None:
    """Embed a single image and upsert to Pinecone."""
    init_index()
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}. Use PNG or JPEG.")

    print(f"Embedding image: {path.name}")
    vector = embed_image(str(path))

    vector_id = f"img-{path.stem}-{uuid.uuid4().hex[:8]}"
    upsert_vector(
        vector_id=vector_id,
        vector=vector,
        metadata={
            "source": str(path),
            "modality": "image",
            "filename": path.name,
            "content_preview": f"[Image: {path.name}]",
        },
        namespace=namespace,
    )
    print(f"Ingested image '{path.name}' → {vector_id}")


def ingest_image_folder(folder_path: str, namespace: str = "") -> int:
    """Ingest all supported images in a folder. Returns count ingested."""
    folder = Path(folder_path)
    images = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        print(f"No supported images found in '{folder}'")
        return 0

    for img_path in images:
        ingest_image_file(str(img_path), namespace=namespace)

    print(f"Ingested {len(images)} images from '{folder.name}'")
    return len(images)
