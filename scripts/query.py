#!/usr/bin/env python3
"""
CLI: Query the Pinecone index using Gemini Embedding 2.

Usage:
  # Text query (retrieval only):
  python scripts/query.py --text "a golden retriever playing in a park"

  # Text query with AI-generated answer (RAG):
  python scripts/query.py --text "what does the report say about Q3 revenue?" --generate

  # Image query:
  python scripts/query.py --image ./data/my_photo.jpg

  # Video query:
  python scripts/query.py --video ./data/my_clip.mp4

  # Filter by modality:
  python scripts/query.py --text "sunset beach" --filter-modality image
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Query Pinecone with Gemini Embedding 2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text query string")
    group.add_argument("--image", help="Path to query image (PNG/JPEG)")
    group.add_argument("--video", help="Path to query video (MP4/MOV)")

    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--namespace", default="", help="Pinecone namespace")
    parser.add_argument(
        "--filter-modality",
        choices=["text", "image", "video"],
        help="Filter results by modality",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate an answer using Claude (requires --text)",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use Gemini instead of Claude for generation",
    )
    args = parser.parse_args()

    if args.generate and not args.text:
        print("Error: --generate requires --text")
        sys.exit(1)

    pinecone_filter = None
    if args.filter_modality:
        pinecone_filter = {"modality": {"$eq": args.filter_modality}}

    if args.generate:
        from src.rag import query_with_answer
        print(f"\nRunning RAG for: \"{args.text}\"\n")
        answer = query_with_answer(
            text=args.text,
            top_k=args.top_k,
            use_claude=not args.use_gemini,
            namespace=args.namespace,
        )
        print("=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
    else:
        from src.rag import query
        results = query(
            text=args.text,
            image_path=args.image,
            video_path=args.video,
            top_k=args.top_k,
            filter=pinecone_filter,
            namespace=args.namespace,
        )

        print(f"\nTop {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"{i}. [{meta.get('modality', '?').upper()}] score={r['score']:.4f}")
            print(f"   source: {meta.get('source', 'N/A')}")
            print(f"   preview: {meta.get('content_preview', '')[:120]}")
            print()


if __name__ == "__main__":
    main()
