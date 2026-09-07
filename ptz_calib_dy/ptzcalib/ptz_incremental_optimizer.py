"""Port of ``src/core/ptz_incremental_optimizer.{h,cc}`` (PTZ-IBA)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .krt_optimizer import FactorType as KRTFactorType
from .krt_optimizer import KRTOptimizer
from .logging_utils import PLOGI
from .ptzray_optimizer import FactorType as RayFactorType
from .ptzray_optimizer import PTZRayOptimizer
from .types import Camera, ImageFeatures, MatchesInfo

LONG_MAX = 9223372036854775807


def _sorted_by_confidence(confidences_rank: np.ndarray) -> List[int]:
    """``sort(indices, rank[A] > rank[B])`` followed by the ``rank <= 0`` cut-off."""
    order = np.argsort(-confidences_rank, kind="stable")
    sorted_image_ids: List[int] = []
    for index in order.tolist():
        if confidences_rank[index] <= 0.0:
            break
        sorted_image_ids.append(int(index))
    return sorted_image_ids


class PtzIncrementalOptimizer:
    """``class PtzIncrementalOptimizer``"""

    kMaxNumImages = 100000
    kBaGlobalImagesRatio = np.float32(1.1)

    def __init__(
        self,
        features: Sequence[ImageFeatures],
        matches_info: Sequence[MatchesInfo],
        cameras: Sequence[Camera],
        max_iter: int,
        names: Sequence[str] = (),
    ):
        self.cameras_ = list(cameras)
        self.features_ = list(features)
        self.matches_info_ = list(matches_info)
        self.names_ = list(names)
        self.max_iter_ = max_iter

        self.init_image_pairs_: Set[int] = set()
        self.num_reg_trials_: Dict[int, int] = {}
        self.reg_image_ids_: Set[int] = set()
        self.seed_image_ids_: List[int] = []

    # ------------------------------------------------------------------
    def num_reg_images(self) -> int:
        return len(self.reg_image_ids_)

    def set_seed_image_id(self, image_ids: Sequence[int]) -> None:
        """``PtzIncrementalOptimizer::SetSeedImageId``"""
        PLOGI("Manually set seed image ids")
        self.seed_image_ids_ = list(image_ids)

    def solve(self) -> Tuple[bool, List[Camera], Set[int]]:
        """``PtzIncrementalOptimizer::Solve``: ``(ok, cameras, reg_image_ids)``.

        The C++ loop falls off the end without a return value when every
        initialisation attempt fails (undefined behaviour); here it is an
        explicit failure.
        """
        if not self._check_valid():
            return False, self.cameras_, set()

        kInitNumTrials = 50
        for _num_trials in range(kInitNumTrials):
            ok, image_id1, image_id2 = self._find_initial_image_pair()
            if not ok:
                PLOGI("No good initial image pair found")
                return False, self.cameras_, set()

            PLOGI(f"Initializing with image pair #{image_id1} and #{image_id2}")

            if not self._register_initial_image_pair(image_id1, image_id2):
                PLOGI(
                    "Initialization failed - possible solutions:\n"
                    "     - try to relax the initialization constraints\n"
                    "     - manually select an initial image pair"
                )
                continue

            PLOGI(
                "Initialization success. Focal: "
                f"{self.cameras_[image_id1].K[0, 0]} , {self.cameras_[image_id2].K[0, 0]}"
            )

            self._adjust_global_bundle()

            ba_prev_num_reg_images = self.num_reg_images()

            reg_next_success = True
            while reg_next_success:
                reg_next_success = False

                next_image_ids = self._find_next_images()
                if not next_image_ids:
                    break

                for reg_trial, image_id in enumerate(next_image_ids):
                    reg_next_success = self._register_next_image(image_id)

                    PLOGI(
                        f"Register image #{image_id}"
                        f"{' success' if reg_next_success else ' failed'}, "
                        f"focal: {self.cameras_[image_id].K[0, 0]}. "
                        f"Currently registered: {self.num_reg_images()}, total: {len(self.features_)}"
                    )

                    if reg_next_success:
                        if self.num_reg_images() >= self.kBaGlobalImagesRatio * ba_prev_num_reg_images:
                            gba_success = self._adjust_global_bundle()
                            if gba_success:
                                ba_prev_num_reg_images = self.num_reg_images()
                                break
                            # TODO(upstream): erase all cameras which did not pass global ba
                            self.reg_image_ids_.discard(image_id)
                            reg_next_success = False

                    if not reg_next_success:
                        PLOGI("Could not register, trying another image")

                        # If the initial pair fails to continue for some time,
                        # abort and try a different initial pair.
                        kMinNumInitialRegTrials = 30
                        kMinModelSize = 3
                        if reg_trial >= kMinNumInitialRegTrials and self.num_reg_images() < kMinModelSize:
                            break

            self._adjust_global_bundle()

            return True, self.cameras_, set(self.reg_image_ids_)

        return False, self.cameras_, set(self.reg_image_ids_)

    # ------------------------------------------------------------------
    def _check_valid(self) -> bool:
        if not self.features_ or len(self.features_) != len(self.cameras_) or self.max_iter_ <= 0:
            return False
        return True

    def _find_initial_image_pair(self) -> Tuple[bool, int, int]:
        """``PtzIncrementalOptimizer::FindInitialImagePair``"""
        if not self.seed_image_ids_:
            image_ids1 = self._find_first_initial_image()
        else:
            image_ids1 = list(self.seed_image_ids_)

        for image_id1 in image_ids1:
            image_ids2 = self._find_second_initial_image(image_id1)

            for image_id2 in image_ids2:
                pair_id = self._image_pair_to_pair_id(image_id1, image_id2)

                # Try every pair only once.
                if pair_id in self.init_image_pairs_:
                    continue

                self.init_image_pairs_.add(pair_id)
                return True, image_id1, image_id2

        return False, LONG_MAX, LONG_MAX

    def _find_first_initial_image(self) -> List[int]:
        """``PtzIncrementalOptimizer::FindFirstInitialImage``"""
        num_imgs = len(self.features_)
        confidences_rank = np.zeros(num_imgs, dtype=np.float32)

        for match_info in self.matches_info_:
            confidences_rank[match_info.src_img_idx] += np.float32(match_info.confidence)
            confidences_rank[match_info.dst_img_idx] += np.float32(match_info.confidence)

        return _sorted_by_confidence(confidences_rank)

    def _find_second_initial_image(self, image_id1: int) -> List[int]:
        """``PtzIncrementalOptimizer::FindSecondInitialImage``"""
        num_imgs = len(self.features_)
        confidences_rank = np.zeros(num_imgs, dtype=np.float32)

        kMinPixelDiff = 50.0
        for match_info in self.matches_info_:
            src_idx = match_info.src_img_idx
            dst_idx = match_info.dst_img_idx
            confidence = np.float32(match_info.confidence)
            matches = match_info.matches

            if matches.shape[0] == 0:
                continue
            if image_id1 != src_idx and image_id1 != dst_idx:
                continue
            if image_id1 == src_idx and image_id1 == dst_idx:
                continue
            if self._cal_pixel_diff(src_idx, dst_idx, matches) < kMinPixelDiff:
                continue
            if image_id1 == src_idx and image_id1 != dst_idx:
                confidences_rank[dst_idx] += confidence
            if image_id1 != src_idx and image_id1 == dst_idx:
                confidences_rank[src_idx] += confidence

        return _sorted_by_confidence(confidences_rank)

    def _find_next_images(self) -> List[int]:
        """``PtzIncrementalOptimizer::FindNextImages``"""
        num_imgs = len(self.features_)
        confidences_rank = np.zeros(num_imgs, dtype=np.float32)

        kMaxRegTrials = 4
        for match_info in self.matches_info_:
            src_idx = match_info.src_img_idx
            dst_idx = match_info.dst_img_idx
            confidence = np.float32(match_info.confidence)

            if src_idx == dst_idx:
                continue
            if match_info.H is None:
                continue

            # Only try registration for a certain maximum number of times.
            if self.num_reg_trials_.get(src_idx, 0) > kMaxRegTrials:
                continue
            if self.num_reg_trials_.get(dst_idx, 0) > kMaxRegTrials:
                continue

            src_reg = src_idx in self.reg_image_ids_
            dst_reg = dst_idx in self.reg_image_ids_
            if src_reg and dst_reg:
                continue  # already registered
            if not src_reg and not dst_reg:
                continue  # not a neighbour
            if src_reg and not dst_reg:
                confidences_rank[dst_idx] += confidence
            elif not src_reg and dst_reg:
                confidences_rank[src_idx] += confidence

        return _sorted_by_confidence(confidences_rank)

    def _cal_pixel_diff(self, image_id1: int, image_id2: int, matches: np.ndarray) -> float:
        """``PtzIncrementalOptimizer::CalPixelDiff``"""
        pt1 = self.features_[image_id1].keypoints[matches[:, 0]]
        pt2 = self.features_[image_id2].keypoints[matches[:, 1]]
        total_dists = float(np.sum(np.linalg.norm(pt1 - pt2, axis=1)))
        return total_dists / matches.shape[0]

    def _image_pair_to_pair_id(self, image_id1: int, image_id2: int) -> int:
        """``PtzIncrementalOptimizer::ImagePairToPairId``"""
        if image_id1 < image_id2:
            return image_id1 * self.kMaxNumImages + image_id2
        return image_id2 * self.kMaxNumImages + image_id1

    def _set_initial_image_pair_parameters(self, image_id1: int, image_id2: int) -> None:
        """``PtzIncrementalOptimizer::SetInitialImagePairParameters``"""
        ratio = 1.2  # this affects fov
        w1, h1 = self.features_[image_id1].img_size
        focal = ratio * max(w1, h1)
        self.cameras_[image_id1].K[0, 0] = focal
        self.cameras_[image_id1].K[1, 1] = focal
        self.cameras_[image_id1].K[0, 2] = 0.5 * w1
        self.cameras_[image_id1].K[1, 2] = 0.5 * h1
        self.cameras_[image_id1].R = np.eye(3)

        w2, h2 = self.features_[image_id2].img_size
        focal = ratio * max(w2, h2)
        self.cameras_[image_id2].K[0, 0] = focal
        self.cameras_[image_id2].K[1, 1] = focal
        self.cameras_[image_id2].K[0, 2] = 0.5 * w2
        self.cameras_[image_id2].K[1, 2] = 0.5 * h2

        for match_info in self.matches_info_:
            i = match_info.src_img_idx
            j = match_info.dst_img_idx
            if i == image_id1 and j == image_id2:
                H_j_i = match_info.H
                if H_j_i is None:
                    break
                R_j_i = np.linalg.inv(self.cameras_[j].K) @ H_j_i @ self.cameras_[i].K
                self.cameras_[j].R = R_j_i @ self.cameras_[i].R
                break

    def _register_initial_image_pair(self, image_id1: int, image_id2: int) -> bool:
        """``PtzIncrementalOptimizer::RegisterInitialImagePair``"""
        self.num_reg_trials_[image_id1] = self.num_reg_trials_.get(image_id1, 0) + 1
        self.num_reg_trials_[image_id2] = self.num_reg_trials_.get(image_id2, 0) + 1

        pair_id = self._image_pair_to_pair_id(image_id1, image_id2)
        self.init_image_pairs_.add(pair_id)

        self._set_initial_image_pair_parameters(image_id1, image_id2)

        optimizer = PTZRayOptimizer(
            self.features_,
            self.matches_info_,
            self.cameras_,
            {image_id1, image_id2},
            self.max_iter_,
            RayFactorType.PTZRay,
        )
        ba_success, _rays = optimizer.solve(self.cameras_)

        if ba_success:
            self.reg_image_ids_.add(image_id1)
            self.reg_image_ids_.add(image_id2)

        return ba_success

    def _register_next_image(self, image_id: int) -> bool:
        """``PtzIncrementalOptimizer::RegisterNextImage``"""
        self.num_reg_trials_[image_id] = self.num_reg_trials_.get(image_id, 0) + 1

        for match_info in self.matches_info_:
            i = match_info.src_img_idx
            j = match_info.dst_img_idx
            H_j_i = match_info.H
            if H_j_i is None:
                continue

            if i in self.reg_image_ids_ and j == image_id:
                self.cameras_[j].K = self.cameras_[i].K
                R_j_i = np.linalg.inv(self.cameras_[j].K) @ H_j_i @ self.cameras_[i].K
                self.cameras_[j].R = R_j_i @ self.cameras_[i].R

                max_iter = 100
                max_reproj_error = 100.0
                optimizer = KRTOptimizer(max_iter, max_reproj_error, KRTFactorType.F)
                optimizer.set_init_params(
                    self.cameras_[j].K, self.cameras_[j].R, self.cameras_[j].t, self.cameras_[j].dist
                )
                optimizer.add_2d2d_constraints(
                    self.cameras_[i],
                    self.features_[i].keypoints,
                    self.features_[j].keypoints,
                    match_info.matches,
                )

                ba_success, K, R, _t, _dist = optimizer.solve()
                if ba_success:
                    self.cameras_[j].K = K
                    self.cameras_[j].R = R
                    self.reg_image_ids_.add(j)
                    return True
                continue

        return False

    def _adjust_global_bundle(self) -> bool:
        """``PtzIncrementalOptimizer::AdjustGlobalBundle``"""
        PLOGI("Global bundle adjustment start")

        optimizer = PTZRayOptimizer(
            self.features_,
            self.matches_info_,
            self.cameras_,
            set(self.reg_image_ids_),
            self.max_iter_,
            RayFactorType.PTZRay,
        )
        ba_success, _rays = optimizer.solve(self.cameras_)
        reprojection_error = optimizer.final_reproj_error_all()

        if ba_success:
            PLOGI(f"Global bundle adjustment success. Rreprojecion error: {reprojection_error}")
        else:
            PLOGI(f"Global bundle adjustment failed. Rreprojecion error: {reprojection_error}")

        return ba_success
