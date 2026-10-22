import io
import unittest

import numpy as np
from PIL import Image

from . import reference_media as reference


def configuration():
    return {
        "image_token_id": 9,
        "text_config": {"vocab_size": 10, "max_position_embeddings": 16},
        "vision_config": {
            "num_hidden_layers": 1,
            "patch_size": 2,
            "downsample_ratio": 1,
            "max_image_tokens": 16,
            "min_pixels": 0,
            "max_wh_ratio": None,
        },
    }


def encoded(color):
    output = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(output, format="PNG")
    return output.getvalue()


class MediaReference(unittest.TestCase):
    def test_expansion_keeps_text_and_whole_images_in_prompt_order(self):
        tokens = np.asarray([1, 9, 2, 9, 3], dtype=np.int64)
        raw = [encoded((255, 0, 0)), encoded((0, 0, 255))]
        result = reference.expand(tokens, raw, configuration())
        np.testing.assert_array_equal(result.tokens, [1, 9, 9, 9, 9, 2, 9, 9, 9, 9, 3])
        np.testing.assert_array_equal(
            result.token_types, [-1, 0, 1, 2, 3, -1, 0, 1, 2, 3, -1]
        )
        self.assertEqual([span.start for span in result.spans], [1, 6])
        np.testing.assert_array_equal(
            result.spans[0].image.patches[0, :, 0, 0], [1, -1, -1]
        )
        np.testing.assert_array_equal(
            result.spans[1].image.patches[0, :, 0, 0], [-1, -1, 1]
        )
        np.testing.assert_array_equal(tokens, [1, 9, 2, 9, 3])
        self.assertFalse(result.tokens.flags.writeable)
        self.assertFalse(result.token_types.flags.writeable)
        self.assertFalse(result.spans[0].image.patches.flags.writeable)
        self.assertFalse(result.spans[0].image.token_types.flags.writeable)

    def test_expansion_rejects_mismatch_and_context_overflow(self):
        raw = encoded((127, 127, 127))
        for tokens, images in (([1, 9], []), ([1], [raw]), ([9], [raw, raw])):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                reference.expand(tokens, images, configuration())
        config = configuration()
        config["text_config"]["max_position_embeddings"] = 4
        full = reference.expand([9], [raw], config)
        self.assertEqual(len(full.tokens), 4)
        with self.assertRaises(ValueError):
            reference.expand([9, 1], [raw], config)
        with self.assertRaises(ValueError):
            reference.expand([1, 2, 3, 4, 5], [], config)
        config["vision_config"]["num_hidden_layers"] = 0
        with self.assertRaises(ValueError):
            reference.expand([9], [raw], config)
        text = reference.expand([1, 2], [], config)
        np.testing.assert_array_equal(text.token_types, [-1, -1])
        self.assertEqual(text.spans, ())
        for tokens in ([], [True], [-1], [10], np.asarray([[1]])):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                reference.expand(tokens, [], config)


if __name__ == "__main__":
    unittest.main()
