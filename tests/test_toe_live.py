from __future__ import annotations

import unittest

from toe_live import compute_preview_shape


class ToeLivePreviewTest(unittest.TestCase):
    def test_preview_size_is_restricted_to_about_quarter_screen_area(self) -> None:
        width, height = compute_preview_shape((1080, 1920))

        self.assertEqual(width, 960)
        self.assertEqual(height, 540)


if __name__ == "__main__":
    unittest.main()
