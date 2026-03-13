"""
Video ingestion: extracts uniformly sampled frames from MP4/MOV videos,
embeds them with Gemini Embedding 2 (up to 6 frames per request),
and upserts to Pinecone.
"""

import uuid
from pathlib import Path

from src.embedder import embed_video_frames
from src.pinecone_client import upsert_vector, init_index

SUPPORTED_EXTENSIONS = {".mp4", ".mov"}


def ingest_video_file(
    file_path: str,
    max_frames: int = 6,
    namespace: str = "",
) -> None:
    """
    Embed a video via frame sampling and upsert to Pinecone.
    max_frames: number of frames to sample (max 6 — model limit).
    """
    init_index()
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format: {path.suffix}. Use MP4 or MOV."
        )

    max_frames = min(max_frames, 6)
    print(f"Embedding video: {path.name} (sampling {max_frames} frames)")

    vector = embed_video_frames(str(path), max_frames=max_frames)

    vector_id = f"vid-{path.stem}-{uuid.uuid4().hex[:8]}"
    upsert_vector(
        vector_id=vector_id,
        vector=vector,
        metadata={
            "source": str(path),
            "modality": "video",
            "filename": path.name,
            "frames_sampled": max_frames,
            "content_preview": f"[Video: {path.name}]",
        },
        namespace=namespace,
    )
    print(f"Ingested video '{path.name}' → {vector_id}")


def ingest_video_folder(
    folder_path: str,
    max_frames: int = 6,
    namespace: str = "",
) -> int:
    """Ingest all supported videos in a folder. Returns count ingested."""
    folder = Path(folder_path)
    videos = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not videos:
        print(f"No supported videos found in '{folder}'")
        return 0

    for vid_path in videos:
        ingest_video_file(str(vid_path), max_frames=max_frames, namespace=namespace)

    print(f"Ingested {len(videos)} videos from '{folder.name}'")
    return len(videos)
