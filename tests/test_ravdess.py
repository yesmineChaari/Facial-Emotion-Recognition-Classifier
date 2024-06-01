import tempfile
import unittest
from pathlib import Path

from src.emotion_recognition.ravdess import discover_videos, parse_ravdess_filename


class RavdessTests(unittest.TestCase):
    def test_parses_emotion_and_metadata(self):
        metadata = parse_ravdess_filename("03-01-05-02-02-01-12.mp4")
        self.assertEqual(metadata.emotion, "angry")
        self.assertEqual(metadata.actor, 12)

    def test_rejects_malformed_filename(self):
        with self.assertRaises(ValueError):
            parse_ravdess_filename("not-a-ravdess-video.mp4")

    def test_discovers_nested_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Actor_01" / "03-01-03-01-01-01-01.mp4"
            path.parent.mkdir()
            path.touch()
            videos = discover_videos(directory)
            self.assertEqual(len(videos), 1)
            self.assertEqual(videos[0].emotion, "happy")


if __name__ == "__main__":
    unittest.main()
