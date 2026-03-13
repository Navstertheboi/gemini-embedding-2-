"""
Gemini Embedding 2 wrapper.
Supports text, images, and video (via frame extraction).
Model: gemini-embedding-2-preview-03-07
SDK: google-genai (google.genai)
"""

import os
import cv2
import numpy as np
import PIL.Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 1536))
MODEL_ID = "gemini-embedding-2-preview-03-07"

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a text string. Use task_type='RETRIEVAL_QUERY' for queries."""
    client = _get_client()
    result = client.models.embed_content(
        model=MODEL_ID,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


def embed_image(image_path: str) -> list[float]:
    """Embed a single image (PNG or JPEG)."""
    client = _get_client()
    image = PIL.Image.open(image_path)
    result = client.models.embed_content(
        model=MODEL_ID,
        contents=image,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


def embed_images(image_paths: list[str]) -> list[float]:
    """
    Embed up to 6 images in a single request (model limit).
    Returns a single unified embedding representing all images.
    """
    if len(image_paths) > 6:
        raise ValueError("Gemini Embedding 2 supports at most 6 images per request.")
    client = _get_client()
    images = [PIL.Image.open(p) for p in image_paths]
    result = client.models.embed_content(
        model=MODEL_ID,
        contents=images,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


def embed_video_frames(video_path: str, max_frames: int = 6) -> list[float]:
    """
    Extract up to max_frames uniformly sampled frames from a video,
    then embed them as a single multimodal request.
    Returns a single embedding representing the video content.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    n = min(max_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n, dtype=int)

    pil_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(PIL.Image.fromarray(frame_rgb))
    cap.release()

    if not pil_frames:
        raise ValueError(f"Could not extract frames from: {video_path}")

    client = _get_client()
    result = client.models.embed_content(
        model=MODEL_ID,
        contents=pil_frames,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


def embed_multimodal(text: str = None, image_paths: list[str] = None) -> list[float]:
    """
    Embed interleaved text + images in a single request.
    Useful for captioned images or document pages with visuals.
    """
    content = []
    if text:
        content.append(text)
    if image_paths:
        for p in image_paths:
            content.append(PIL.Image.open(p))
    if not content:
        raise ValueError("Provide at least one of text or image_paths.")
    client = _get_client()
    result = client.models.embed_content(
        model=MODEL_ID,
        contents=content,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values
