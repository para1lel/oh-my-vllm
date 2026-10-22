import copy
import unittest
from dataclasses import FrozenInstanceError

from . import reference_video as reference


class VideoReference(unittest.TestCase):
    def test_sample_plan_keeps_endpoints_rounding_and_frame_limit(self):
        cases = (
            (1, 30, (0,)),
            (3, 30, (0, 1, 2)),
            (6, 30, (0, 2, 3, 5)),
            (8, 4, (0, 2, 5, 7)),
            (11, 4, (0, 2, 5, 8, 10)),
            (12, 4, (0, 2, 4, 7, 9, 11)),
            (4, 4e-308, (0, 1, 2, 3)),
            (
                2**53,
                2**53,
                (0, 3002399751580330, 6004799503160661, 9007199254740991),
            ),
        )
        for total, fps, expected in cases:
            with self.subTest(total=total, fps=fps):
                self.assertEqual(
                    reference.sample_plan(reference.VideoMetadata(total, fps)),
                    expected,
                )
        capped = reference.sample_plan(reference.VideoMetadata(770, 2))
        self.assertEqual(len(capped), 768)
        self.assertEqual(capped[:3], (0, 1, 2))
        self.assertEqual(capped[-3:], (767, 768, 769))
        self.assertEqual(tuple(sorted(set(capped))), capped)

    def test_expansion_keeps_frame_and_surrounding_media_order_at_timestamp_ties(self):
        metadata = reference.VideoMetadata(6, 30)
        images = (
            {"type": "image", "data": b"frame-zero"},
            {"type": "image", "source": {"data": "frame-two"}},
            {"type": "image", "url": "fixture://frame-three"},
            {"type": "image", "data": b"frame-five"},
        )
        frames = tuple(zip((0, 2, 3, 5), images, strict=True))
        original = copy.deepcopy(frames)
        before = (
            {"type": "text", "text": "Compare this still and video."},
            {"type": "image", "url": "fixture://still"},
        )
        after = ({"type": "text", "text": "Describe the change."},)
        expanded = reference.video_records(metadata, frames)
        self.assertEqual(
            before + expanded + after,
            (
                *before,
                {"type": "text", "text": "<0.0 seconds>"},
                images[0],
                {"type": "text", "text": "<0.1 seconds>"},
                images[1],
                {"type": "text", "text": "<0.1 seconds>"},
                images[2],
                {"type": "text", "text": "<0.2 seconds>"},
                images[3],
                *after,
            ),
        )
        self.assertEqual(frames, original)
        self.assertIsNot(expanded[1], images[0])
        odd_frames = ((0, images[0]), (1, images[1]), (2, images[2]))
        odd = reference.video_records(reference.VideoMetadata(3, 100), odd_frames)
        self.assertEqual(odd[::2], ({"type": "text", "text": "<0.0 seconds>"},) * 3)
        self.assertEqual(odd[1::2], images[:3])

    def test_rejects_invalid_metadata_and_mismatched_frame_records(self):
        for total in (True, False, 0, -1, 1.0, "3", None):
            with self.subTest(total=total), self.assertRaises(ValueError):
                reference.VideoMetadata(total, 30)
        for total, fps in (
            (2**53 + 1, 30),
            (9007199254740996, 9007199254740996),
            (10**309, 30),
        ):
            with self.subTest(total=total, fps=fps), self.assertRaises(ValueError):
                reference.VideoMetadata(total, fps)
        for fps in (True, False, 0, -1, float("nan"), float("inf"), "30", None):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                reference.VideoMetadata(3, fps)
        metadata = reference.VideoMetadata(6, 30)
        with self.assertRaises(FrozenInstanceError):
            metadata.frame_count = 1
        image = {"type": "image", "data": b"frame"}
        frames = tuple((index, image) for index in (0, 2, 3, 5))
        bad_frames = (
            frames[:-1],
            (*frames, (5, image)),
            frames[::-1],
            (frames[0], frames[1], frames[1], frames[3]),
            ((False, image), *frames[1:]),
            ((0.0, image), *frames[1:]),
            ((0, {"type": "text", "text": "frame"}), *frames[1:]),
            ((0, {"type": "image"}), *frames[1:]),
            ((0, {"type": "image", "url": ""}), *frames[1:]),
            ((0,), *frames[1:]),
        )
        for records in bad_frames:
            with self.subTest(records=records), self.assertRaises(ValueError):
                reference.video_records(metadata, records)
        with self.assertRaises(ValueError):
            reference.video_records(
                reference.VideoMetadata(2, 5e-324), ((0, image), (1, image))
            )
        with self.assertRaises(ValueError):
            reference.sample_plan({"frame_count": 6, "fps": 30})


if __name__ == "__main__":
    unittest.main()
