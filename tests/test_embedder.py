"""
Unit tests for src/embedder.py.
Uses mocking so no real API calls are made.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

DIMS = 1536
FAKE_EMBEDDING = [0.1] * DIMS


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("PINECONE_API_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", str(DIMS))


@pytest.fixture
def mock_genai():
    with patch("src.embedder.genai") as mock:
        mock.embed_content.return_value = {"embedding": FAKE_EMBEDDING}
        yield mock


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestEmbedText:
    def test_returns_list_of_floats(self, mock_genai):
        from src.embedder import embed_text
        result = embed_text("hello world")
        assert isinstance(result, list)
        assert len(result) == DIMS
        assert all(isinstance(v, float) for v in result)

    def test_calls_embed_content_with_correct_model(self, mock_genai):
        from src.embedder import embed_text
        embed_text("test query", task_type="RETRIEVAL_QUERY")
        call_kwargs = mock_genai.embed_content.call_args
        assert call_kwargs.kwargs["model"] == "models/gemini-embedding-2-preview"
        assert call_kwargs.kwargs["task_type"] == "RETRIEVAL_QUERY"

    def test_uses_configured_dimensions(self, mock_genai):
        from src.embedder import embed_text
        embed_text("test")
        call_kwargs = mock_genai.embed_content.call_args
        assert call_kwargs.kwargs["output_dimensionality"] == DIMS


class TestEmbedImage:
    def test_returns_embedding(self, mock_genai, tmp_path):
        # Create a minimal 1x1 PNG
        import PIL.Image
        img_path = tmp_path / "test.png"
        PIL.Image.new("RGB", (1, 1), color=(255, 0, 0)).save(img_path)

        with patch("src.embedder.PIL.Image.open") as mock_open:
            mock_open.return_value = MagicMock()
            from src.embedder import embed_image
            result = embed_image(str(img_path))

        assert isinstance(result, list)
        assert len(result) == DIMS

    def test_unsupported_image_raises_on_open(self, mock_genai, tmp_path):
        # PIL.Image.open would raise for a bad file; we just verify it propagates
        bad_path = tmp_path / "test.gif"
        bad_path.write_bytes(b"GIF89a")
        with patch("src.embedder.PIL.Image.open", side_effect=Exception("bad format")):
            from src.embedder import embed_image
            with pytest.raises(Exception, match="bad format"):
                embed_image(str(bad_path))


class TestEmbedImages:
    def test_raises_for_more_than_6_images(self, mock_genai):
        from src.embedder import embed_images
        with pytest.raises(ValueError, match="at most 6 images"):
            embed_images(["img.jpg"] * 7)

    def test_accepts_up_to_6(self, mock_genai):
        with patch("src.embedder.PIL.Image.open", return_value=MagicMock()):
            from src.embedder import embed_images
            result = embed_images(["img.jpg"] * 6)
        assert len(result) == DIMS


class TestEmbedVideoFrames:
    def test_raises_for_invalid_video(self, mock_genai, tmp_path):
        bad_video = tmp_path / "bad.mp4"
        bad_video.write_bytes(b"not a video")
        with patch("src.embedder.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = False
            from src.embedder import embed_video_frames
            with pytest.raises(ValueError, match="Cannot open video"):
                embed_video_frames(str(bad_video))

    def test_samples_correct_number_of_frames(self, mock_genai, tmp_path):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.embedder.cv2.VideoCapture") as mock_cap:
            instance = mock_cap.return_value
            instance.isOpened.return_value = True
            instance.get.return_value = 100  # 100 total frames
            instance.read.return_value = (True, fake_frame)

            from src.embedder import embed_video_frames
            embed_video_frames(str(tmp_path / "v.mp4"), max_frames=4)

            call_args = mock_genai.embed_content.call_args
            frames_passed = call_args.kwargs["content"]
            assert len(frames_passed) == 4
