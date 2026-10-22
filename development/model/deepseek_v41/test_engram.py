import unittest

import numpy as np
from tokenizers import Tokenizer, decoders, models

from .engram import HashLayout, build_layout, compressed_token_map, hash_rows


class EngramReference(unittest.TestCase):
    def test_normalization_merges_text_but_keeps_space_and_empty(self):
        vocabulary = {"THE": 0, "the": 1, " Th\u00e9 ": 2, " ": 3, "": 4, "A": 5}
        tokenizer = Tokenizer(models.WordLevel(vocabulary))
        mapping, size = compressed_token_map(tokenizer)
        np.testing.assert_array_equal(mapping, [0, 0, 0, 1, 2, 3])
        self.assertEqual(size, 4)
        partial_bytes = Tokenizer(
            models.BPE({"\u00c3": 0, "\u00a9": 1, "a": 2, "A": 3}, merges=[])
        )
        partial_bytes.decoder = decoders.ByteLevel()
        mapping, size = compressed_token_map(partial_bytes)
        np.testing.assert_array_equal(mapping, [0, 1, 2, 2])
        self.assertEqual(size, 3)

    def test_layout_keeps_disjoint_prime_buckets_and_bounded_products(self):
        layout = build_layout(
            layer_ids=(0, 1),
            max_ngram_size=3,
            n_heads=2,
            bucket_size=5,
            compressed_vocab_size=10,
            table_sizes=(36, 88),
        )
        self.assertEqual(layout.primes, ((5, 7, 11, 13), (17, 19, 23, 29)))
        self.assertEqual(layout.offsets, ((0, 5, 12, 23), (0, 17, 36, 59)))
        self.assertEqual(
            layout.multipliers,
            (
                (587493461538853713, 248834323164462581, 37791405492449801),
                (759362766141463323, 47959300183862325, 356331584946896373),
            ),
        )
        for row in layout.multipliers:
            for multiplier in row:
                self.assertEqual(multiplier % 2, 1)
                self.assertLessEqual(multiplier * 9, (1 << 63) - 1)
        with self.assertRaisesRegex(ValueError, "table"):
            build_layout(
                layer_ids=(0,),
                max_ngram_size=3,
                n_heads=2,
                bucket_size=5,
                compressed_vocab_size=10,
                table_sizes=(35,),
            )

    def test_dead_token_blocks_lookback_and_chunking_keeps_history(self):
        layout = HashLayout(
            max_ngram_size=3,
            n_heads=1,
            vocabulary_size=10,
            primes=((5, 7),),
            offsets=((0, 5),),
            multipliers=((1, 3, 5),),
        )
        # At the dead token all terms use pad=2; the next row cannot see token 4.
        expected = np.array([[2, 11], [2, 11], [4, 5], [0, 6], [1, 6]])[:, None, :]
        tokens = [1, 4, -1, 3, 2]
        actual, tail = hash_rows(tokens, layout, pad_id=np.int64(2))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(tail, (3, 2))
        for split in range(len(tokens) + 1):
            first, history = hash_rows(tokens[:split], layout, pad_id=2)
            second, end = hash_rows(tokens[split:], layout, pad_id=2, history=history)
            np.testing.assert_array_equal(np.concatenate((first, second)), expected)
            self.assertEqual(end, tail)
        for invalid in ([-2], [10], [True], [1.5]):
            with self.subTest(tokens=invalid), self.assertRaises(ValueError):
                hash_rows(invalid, layout, pad_id=2)
        for invalid in (-1, 10, True, np.bool_(True), 1.5):
            with self.subTest(pad_id=invalid), self.assertRaises(ValueError):
                hash_rows(tokens, layout, pad_id=invalid)
