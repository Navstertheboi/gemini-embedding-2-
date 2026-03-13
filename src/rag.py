"""
Multimodal RAG query pipeline.

Flow:
  1. Embed the user query (text or image or video)
  2. Search Pinecone for nearest neighbors
  3. Return ranked results with metadata
  4. Optionally generate a final answer via Gemini or Claude
"""

import os
from dotenv import load_dotenv

from src.embedder import embed_text, embed_image, embed_video_frames
from src.pinecone_client import query_vectors, init_index

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def query(
    text: str = None,
    image_path: str = None,
    video_path: str = None,
    top_k: int = 5,
    filter: dict = None,
    namespace: str = "",
) -> list[dict]:
    """
    Embed a query and retrieve the top_k most similar items from Pinecone.

    Provide exactly one of: text, image_path, or video_path.

    Returns a list of dicts with keys: id, score, metadata.
    """
    init_index()

    inputs = [x for x in [text, image_path, video_path] if x is not None]
    if len(inputs) != 1:
        raise ValueError("Provide exactly one of: text, image_path, or video_path.")

    if text:
        query_vector = embed_text(text, task_type="RETRIEVAL_QUERY")
        modality = "text"
    elif image_path:
        query_vector = embed_image(image_path)
        modality = "image"
    else:
        query_vector = embed_video_frames(video_path)
        modality = "video"

    print(f"Querying Pinecone with {modality} embedding (top_k={top_k})...")
    results = query_vectors(
        vector=query_vector,
        top_k=top_k,
        filter=filter,
        namespace=namespace,
    )

    return results


def query_with_answer(
    text: str,
    top_k: int = 5,
    use_claude: bool = False,
    namespace: str = "",
) -> str:
    """
    RAG with generation: retrieve context then generate an answer.
    Uses Claude (Anthropic) by default; falls back to Gemini Pro.
    Only supports text queries for the generation step.
    """
    results = query(text=text, top_k=top_k, namespace=namespace)

    context_parts = []
    for r in results:
        meta = r["metadata"]
        modality = meta.get("modality", "unknown")
        preview = meta.get("content_preview", "")
        source = meta.get("source", "")
        context_parts.append(
            f"[{modality.upper()} | source: {source}]\n{preview}"
        )

    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        f"Use the following retrieved context to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {text}\n\nAnswer:"
    )

    if use_claude and ANTHROPIC_API_KEY:
        return _generate_with_claude(prompt)
    else:
        return _generate_with_gemini(prompt)


def _generate_with_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _generate_with_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text
