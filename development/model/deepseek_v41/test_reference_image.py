import io
import unittest

import numpy as np
from PIL import Image

from . import reference_image as reference


class ImageReference(unittest.TestCase):
    def test_grid_uses_source_minimum_and_token_bound_not_qwen_rounding(self):
        config = {
            "patch_size": 14,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        }
        square = reference.plan(1, 1, config)
        self.assertEqual((square.height, square.width), (546, 546))
        self.assertEqual((square.patch_h, square.patch_w), (39, 39))
        self.assertEqual(
            (square.aligned_h, square.aligned_w, square.tokens), (13, 13, 184)
        )
        reduced = reference.plan(2048, 1024, config)
        self.assertEqual((reduced.height, reduced.width), (924, 1848))
        self.assertEqual(
            (reduced.aligned_h, reduced.aligned_w, reduced.tokens), (22, 44, 992)
        )
        tall = reference.plan(1, 10000, config)
        wide = reference.plan(10000, 1, config)
        self.assertEqual((tall.aligned_h, tall.aligned_w), (511, 1))
        self.assertEqual((wide.aligned_h, wide.aligned_w), (1, 1021))
        for dimensions in ((0, 1), (1, -1), (True, 2)):
            with self.assertRaises(ValueError):
                reference.plan(*dimensions, config)

    def test_pixels_keep_rgb_patch_order_and_float64_normalization(self):
        config = {
            "patch_size": 2,
            "downsample_ratio": 1,
            "max_image_tokens": 16,
            "min_pixels": 0,
            "max_wh_ratio": None,
        }
        pixels = np.asarray(
            [[[0, 127, 255], [255, 128, 0], [128, 0, 127], [127, 255, 128]]] * 2,
            dtype=np.uint8,
        )
        data = io.BytesIO()
        Image.fromarray(pixels).save(data, format="PNG")
        result = reference.prepare(data.getvalue(), config)
        self.assertEqual(result.patches.dtype, np.float64)
        self.assertEqual(result.patches.shape, (2, 3, 2, 2))
        np.testing.assert_allclose(
            result.patches[0, :, 0, 0], [-1, -1 / 255, 1], atol=1e-15
        )
        np.testing.assert_allclose(
            result.patches[0, :, 0, 1], [1, 1 / 255, -1], atol=1e-15
        )
        np.testing.assert_allclose(
            result.patches[1, :, 0, 0], [1 / 255, -1, -1 / 255], atol=1e-15
        )
        np.testing.assert_array_equal(result.token_types, [0, 1, 1, 2, 3])
        np.testing.assert_array_equal(pixels[0, 0], [0, 127, 255])
        for raw in (b"not an image", b"", "a path is not image bytes"):
            with self.assertRaises((ValueError, TypeError)):
                reference.prepare(raw, config)

    def test_padding_and_wide_resize_follow_source_branches(self):
        config = {
            "patch_size": 2,
            "downsample_ratio": 1,
            "max_image_tokens": 32,
            "min_pixels": 0,
            "max_wh_ratio": None,
        }
        source = Image.new("RGB", (2, 1), (255, 0, 0))
        raw = io.BytesIO()
        source.save(raw, format="PNG")
        padded = reference.prepare(raw.getvalue(), config)
        np.testing.assert_allclose(padded.patches[0, :, 0, 0], [1, -1, -1], atol=0)
        np.testing.assert_allclose(
            padded.patches[0, :, 1, 0], [-1 / 255] * 3, atol=1e-15
        )
        clipped = reference.prepare(raw.getvalue(), dict(config, max_wh_ratio=1))
        self.assertEqual((clipped.plan.width, clipped.plan.height), (2, 2))
        np.testing.assert_allclose(clipped.patches[0, :, 1, 0], [1, -1, -1], atol=0)
        np.testing.assert_array_equal(clipped.token_types, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
