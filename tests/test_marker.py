from __future__ import annotations

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from marker import detect_bb_near_anchor


class BBMarkerDetectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = (80.0, 120.0)
        self.args = SimpleNamespace(
            bb_search_radius=80.0,
            blur_kernel=5,
            morph_kernel=3,
            min_area=10.0,
            bb_max_area=500.0,
            bb_target_area=200.0,
            bb_max_motion=40.0,
            bb_min_circularity=0.6,
            bb_min_aspect=0.6,
        )

    @staticmethod
    def blank_frame() -> np.ndarray:
        return np.full((240, 320, 3), 25, dtype=np.uint8)

    def test_selects_marker_near_anchor_instead_of_red_distractors(self) -> None:
        frame = self.blank_frame()
        cv2.circle(frame, (80, 120), 8, (0, 0, 255), -1)
        cv2.circle(frame, (132, 120), 8, (0, 0, 255), -1)
        cv2.circle(frame, (260, 70), 13, (0, 0, 255), -1)

        detection, _ = detect_bb_near_anchor(frame, self.anchor, self.args)

        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertAlmostEqual(detection.x, 80.0, delta=0.2)
        self.assertAlmostEqual(detection.y, 120.0, delta=0.2)

    def test_rejects_distractor_when_marker_is_occluded(self) -> None:
        frame = self.blank_frame()
        cv2.circle(frame, (132, 120), 8, (0, 0, 255), -1)
        cv2.circle(frame, (260, 70), 13, (0, 0, 255), -1)

        detection, _ = detect_bb_near_anchor(frame, self.anchor, self.args)

        self.assertIsNone(detection)

    def test_rejects_non_circular_red_region_near_anchor(self) -> None:
        frame = self.blank_frame()
        cv2.rectangle(frame, (68, 117), (92, 123), (0, 0, 255), -1)

        detection, _ = detect_bb_near_anchor(frame, self.anchor, self.args)

        self.assertIsNone(detection)


if __name__ == "__main__":
    unittest.main()
