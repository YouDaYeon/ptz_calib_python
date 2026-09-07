#!/usr/bin/env python
"""Port of ``src/app/run_ptz_ba.{h,cc}`` - PTZ-IBA + georeferencing (offline stage)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Sequence, Set, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptzcalib import data_io  # noqa: E402
from ptzcalib.logging_utils import PLOGE, PLOGI, init_logging  # noqa: E402
from ptzcalib.os_path import basename, mkdir_ifnot_exist  # noqa: E402
from ptzcalib.ptz_incremental_optimizer import PtzIncrementalOptimizer  # noqa: E402
from ptzcalib.ptzray_optimizer import FactorType, PTZRayOptimizer  # noqa: E402
from ptzcalib.types import Camera, ImageFeatures, MatchesInfo  # noqa: E402

MAX_ITER = 200


def parse_args() -> argparse.Namespace:
    """``ParseArgs``"""
    parser = argparse.ArgumentParser(description="PTZ-Calib offline stage (PTZ-BA)")
    parser.add_argument("-i", "--images", required=True, help="Images directory")
    parser.add_argument("-f", "--features", required=True, help="Features and matches directory")
    parser.add_argument("-a", "--annotation", default="", help="Annotation filepath")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--no_georef",
        action="store_true",
        help="Skip georeferencing; write PTZ-IBA parameters in the local frame (no survey data needed)",
    )
    parser.add_argument("--dist", action="store_true", help="Whether images have distortion")
    parser.add_argument("--seed", type=int, default=0, help="OpenCV RNG seed (RANSAC reproducibility)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def run_ptz_ba(
    fnames: Sequence[str],
    features: Sequence[ImageFeatures],
    matches_info: Sequence[MatchesInfo],
    max_iter: int,
) -> Tuple[bool, List[Camera], Set[int]]:
    """``RunPtzBA``"""
    num_images = len(fnames)
    cameras = [Camera() for _ in range(num_images)]

    ptz_iba = PtzIncrementalOptimizer(features, matches_info, cameras, max_iter, fnames)
    ba_success, cameras, reg_image_ids = ptz_iba.solve()

    return ba_success, cameras, reg_image_ids


def run_georeferencing(
    features: Sequence[ImageFeatures],
    matches_info: Sequence[MatchesInfo],
    pixels: Sequence[np.ndarray],
    pts3d: Sequence[np.ndarray],
    cam_ids: Set[int],
    max_iter: int,
    has_dist: bool,
    cameras: List[Camera],
) -> Tuple[bool, float, float]:
    """``RunGeoreferencing``"""
    factor_type = FactorType.PTZRayDist if has_dist else FactorType.PTZRay

    optimizer = PTZRayOptimizer(
        features, matches_info, cameras, cam_ids, max_iter, factor_type, pixels=pixels, pts3d=pts3d
    )
    ba_success, _rays = optimizer.solve(cameras)
    if not ba_success:
        return False, -1.0, -1.0

    return True, optimizer.final_reproj_error_2d2d(), optimizer.final_reproj_error_2d3d()


def main() -> int:
    args = parse_args()
    init_logging(args.verbose)
    cv2.setRNGSeed(args.seed)

    t_start = time.time()

    load_status, fnames, features, _sizes = data_io.load_imgs_and_features(args.images, args.features)
    if not load_status:
        PLOGE("Error loading images and features. Exiting ...")
        return -1

    matches_path = args.features + "/pairs_matches.txt"
    load_status, matches_info = data_io.load_matches_info(matches_path, fnames, features)
    if not load_status:
        PLOGE(f"Error loading matches from {matches_path}. Exiting ...")
        return -1
    PLOGI("================== PTZ-IBA Begin ==========================")

    ba_status, cameras, reg_image_ids = run_ptz_ba(fnames, features, matches_info, MAX_ITER)
    if not ba_status:
        PLOGI("================== PTZ-IBA End: failed ==========================")
        return -1
    PLOGI("================== PTZ-IBA End: success ==========================")

    if args.no_georef:
        # save_registered_cam handles empty pixels/pts3d (run_ptz_reloc.py already relies on it)
        pixels = [np.zeros((0, 2)) for _ in fnames]
        pts3d = [np.zeros((0, 3)) for _ in fnames]
        error_2d2d, error_2d3d = -1.0, -1.0
        PLOGI("================== Georeferencing skipped (--no_georef) ==================")
    else:
        load_status, pixels, pts3d = data_io.load_annotation(args.annotation, fnames)
        if not load_status:
            PLOGE(f"Error loading annotation from {args.annotation}. Exiting ...")
            return -1
        PLOGI(f"Success loading annotation from {args.annotation}")
        PLOGI("================== Georeferencing Begin ==========================")

        georef_status, error_2d2d, error_2d3d = run_georeferencing(
            features, matches_info, pixels, pts3d, reg_image_ids, MAX_ITER, args.dist, cameras
        )
        if not georef_status:
            PLOGI("================== Georeferencing End: failed ==========================")
            return -1
        PLOGI("================== Georeferencing End: success ==========================")

    cam_id = basename(args.images)
    out_dir = args.output
    mkdir_ifnot_exist(out_dir)
    out_path = out_dir + "/" + cam_id + ".json"

    data_io.save_registered_cam(cameras, reg_image_ids, fnames, pixels, pts3d, out_path)

    PLOGI("================== Summary Begin ==========================")
    PLOGI(f"Registered/Total: {len(reg_image_ids)}/{len(fnames)}")
    PLOGI(f"Error 2d-2d: {error_2d2d}")
    PLOGI(f"Error 2d-3d: {error_2d3d}")
    PLOGI(f"Elapsed: {time.time() - t_start:.1f}s")
    PLOGI("==================== Summary End ==========================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
