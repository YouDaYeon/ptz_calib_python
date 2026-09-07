#!/usr/bin/env python
"""Port of ``src/app/run_ptz_reloc.{h,cc}`` - PTZ-Reloc (online stage)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptzcalib import data_io  # noqa: E402
from ptzcalib.krt_optimizer import FactorType, KRTOptimizer  # noqa: E402
from ptzcalib.logging_utils import PLOGE, PLOGI, PLOGW, init_logging  # noqa: E402
from ptzcalib.os_path import basename, mkdir_ifnot_exist  # noqa: E402
from ptzcalib.types import Camera  # noqa: E402

MAX_ITER = 200
MAX_REPROJ_ERROR = 100.0

BestMatchT = Tuple[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    """``ParseArgs``"""
    parser = argparse.ArgumentParser(description="PTZ-Calib online stage (PTZ-Reloc)")
    parser.add_argument("--ref_images", required=True, help="Reference images directory")
    parser.add_argument("--ref_features", required=True, help="Reference images features directory")
    parser.add_argument("--ref_params", required=True, help="Reference camera parameters filepath")
    parser.add_argument("--test_images", required=True, help="Test images directory")
    parser.add_argument("--test_features", required=True, help="Test images features and matches directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--dist", action="store_true", help="Whether images have distortion")
    parser.add_argument("--seed", type=int, default=0, help="OpenCV RNG seed")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def find_best_match(
    fname: str,
    img_pairs_name: Sequence[Tuple[str, str]],
    pairs_matches: Sequence[np.ndarray],
) -> BestMatchT:
    """``FindBestMatch``: reference image with the largest number of matches."""
    best_name = ""
    best_matches = np.zeros((0, 2), dtype=np.int32)

    for i in range(len(img_pairs_name)):
        if img_pairs_name[i][1] != fname:
            continue
        if pairs_matches[i].shape[0] > best_matches.shape[0]:
            best_name = img_pairs_name[i][0]
            best_matches = pairs_matches[i]

    return best_name, best_matches


def main() -> int:
    args = parse_args()
    init_logging(args.verbose)
    cv2.setRNGSeed(args.seed)

    t_start = time.time()

    load_status, ref_fnames, ref_features, _ref_sizes = data_io.load_imgs_and_features(
        args.ref_images, args.ref_features
    )
    if not load_status:
        PLOGE("Error loading reference images and features. Exiting ...")
        return -1

    load_status, test_fnames, test_features, test_sizes = data_io.load_imgs_and_features(
        args.test_images, args.test_features
    )
    if not load_status:
        PLOGE("Error loading test images and features. Exiting ...")
        return -1

    matches_path = args.test_features + "/pairs_matches.txt"
    pairs_matches, img_pairs_name = data_io.read_colmap_matches(matches_path)

    load_status, ref_cameras = data_io.read_cam_from_json(args.ref_params, ref_fnames)
    if not load_status:
        PLOGE("Error loading reference camera parameters. Exiting ...")
        return -1

    test_cameras = [Camera() for _ in test_fnames]
    success_ids: Set[int] = set()

    for test_idx in range(len(test_fnames)):
        best_name, best_matches = find_best_match(test_fnames[test_idx], img_pairs_name, pairs_matches)
        ref_idx = data_io.find_img_index(ref_fnames, best_name)
        if ref_idx == -1:
            PLOGI(f"Running ptz-reloc failed: {test_fnames[test_idx]}")
            continue

        ref_cam = ref_cameras[ref_idx]

        factor_type = FactorType.FDist if args.dist else FactorType.F
        optimizer = KRTOptimizer(MAX_ITER, MAX_REPROJ_ERROR, factor_type)

        # Set initial parameters
        f = ref_cam.K[0, 0]
        cx = 0.5 * test_sizes[test_idx][0]
        cy = 0.5 * test_sizes[test_idx][1]
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        R = ref_cam.R
        t = ref_cam.t
        dist = ref_cam.dist
        optimizer.set_init_params(K, R, t, dist)

        optimizer.add_2d2d_constraints(
            ref_cam, ref_features[ref_idx].keypoints, test_features[test_idx].keypoints, best_matches
        )

        opti_success, K, R, t, dist = optimizer.solve()

        if opti_success:
            test_cameras[test_idx] = Camera(K, R, t, dist)
            success_ids.add(test_idx)
            PLOGI(f"Running ptz-reloc success: {test_fnames[test_idx]}")
        else:
            PLOGI(f"Running ptz-reloc failed: {test_fnames[test_idx]}")

    cam_id = basename(args.test_images)
    out_dir = args.output
    mkdir_ifnot_exist(out_dir)
    out_path = out_dir + "/" + cam_id + ".json"

    pixels = [np.zeros((0, 2)) for _ in test_fnames]
    pts3d = [np.zeros((0, 3)) for _ in test_fnames]
    data_io.save_registered_cam(test_cameras, success_ids, test_fnames, pixels, pts3d, out_path)

    elapsed = time.time() - t_start
    n = max(len(test_fnames), 1)
    PLOGI(
        f"PTZ-Reloc done: {len(success_ids)}/{len(test_fnames)} succeeded, "
        f"{elapsed:.1f}s total ({elapsed / n:.2f}s per image)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
