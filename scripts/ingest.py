#!/usr/bin/env python3
"""
CLI: Ingest text, image, or video files into Pinecone.

Usage:
  python scripts/ingest.py --path ./data/doc.txt
  python scripts/ingest.py --path ./data/photo.jpg
  python scripts/ingest.py --path ./data/clip.mp4
  python scripts/ingest.py --path ./data/images/   # folder of images
  python scripts/ingest.py --path ./data/videos/   # folder of videos
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}


def main():
    parser = argparse.ArgumentParser(description="Ingest files into Pinecone via Gemini Embedding 2")
    parser.add_argument("--path", required=True, help="File or folder to ingest")
    parser.add_argument("--namespace", default="", help="Pinecone namespace (optional)")
    parser.add_argument("--max-frames", type=int, default=6, help="Max frames to sample from videos (1-6)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: path does not exist: {path}")
        sys.exit(1)

    if path.is_dir():
        # Auto-detect folder contents
        from src.ingestion.image_ingest import ingest_image_folder
        from src.ingestion.video_ingest import ingest_video_folder
        print(f"Ingesting folder: {path}")
        ingest_image_folder(str(path), namespace=args.namespace)
        ingest_video_folder(str(path), max_frames=args.max_frames, namespace=args.namespace)

    elif path.suffix.lower() in TEXT_EXTENSIONS:
        from src.ingestion.text_ingest import ingest_text_file
        ingest_text_file(str(path), namespace=args.namespace)

    elif path.suffix.lower() in IMAGE_EXTENSIONS:
        from src.ingestion.image_ingest import ingest_image_file
        ingest_image_file(str(path), namespace=args.namespace)

    elif path.suffix.lower() in VIDEO_EXTENSIONS:
        from src.ingestion.video_ingest import ingest_video_file
        ingest_video_file(str(path), max_frames=args.max_frames, namespace=args.namespace)

    else:
        print(f"Unsupported file type: {path.suffix}")
        print(f"Supported: {TEXT_EXTENSIONS | IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}")
        sys.exit(1)


if __name__ == "__main__":
    main()
