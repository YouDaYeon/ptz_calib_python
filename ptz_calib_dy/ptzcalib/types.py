"""Port of ``src/core/types.{h,cc}``.

``cv::KeyPoint``/``cv::DMatch`` vectors are replaced by numpy arrays:

* ``ImageFeatures.keypoints`` -> ``(N, 2) float64`` array of ``kpt.pt``
* ``MatchesInfo.matches``     -> ``(M, 2) int32`` array of ``(queryIdx, trainIdx)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ImageFeatures:
    """``struct ImageFeatures``"""

    img_idx: int = -1
    img_size: Tuple[int, int] = (0, 0)  # (width, height), as cv::Size
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    descriptors: Optional[np.ndarray] = None


@dataclass
class MatchesInfo:
    """``struct MatchesInfo``

    A default-constructed instance mirrors the value-initialised C++ struct that
    ``LoadMatchesInfo`` puts into the ``num_images * num_images`` table.
    """

    src_img_idx: int = 0
    dst_img_idx: int = 0
    matches: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    inliers_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint8))
    num_inliers: int = 0
    H: Optional[np.ndarray] = None  # cv::Mat H (empty -> None)
    confidence: float = 0.0


@dataclass
class Ray:
    """``struct Ray``"""

    id_: int
    pt3d_: np.ndarray  # (3,)
    uv_: np.ndarray  # (2,)


class Camera:
    """``class Camera``: K (3x3), R (3x3, world->cam), t (3x1), dist (5x1)."""

    __slots__ = ("K_", "R_", "t_", "dist_")

    def __init__(self, K=None, R=None, t=None, dist=None):
        if K is None and R is None and t is None and dist is None:
            self.K_ = np.eye(3, dtype=np.float64)
            self.R_ = np.eye(3, dtype=np.float64)
            self.t_ = np.zeros((3, 1), dtype=np.float64)
            self.dist_ = np.zeros((5, 1), dtype=np.float64)
        else:
            self.K_ = np.array(K, dtype=np.float64).reshape(3, 3).copy()
            self.R_ = np.array(R, dtype=np.float64).reshape(3, 3).copy()
            self.t_ = np.array(t, dtype=np.float64).reshape(3, 1).copy()
            self.dist_ = np.array(dist, dtype=np.float64).reshape(5, 1).copy()

    # --- accessors (C++ K()/R()/t()/dist()) -------------------------------
    @property
    def K(self) -> np.ndarray:
        return self.K_

    @K.setter
    def K(self, value: np.ndarray) -> None:
        self.K_ = np.array(value, dtype=np.float64).reshape(3, 3).copy()

    @property
    def R(self) -> np.ndarray:
        return self.R_

    @R.setter
    def R(self, value: np.ndarray) -> None:
        self.R_ = np.array(value, dtype=np.float64).reshape(3, 3).copy()

    @property
    def t(self) -> np.ndarray:
        return self.t_

    @t.setter
    def t(self, value: np.ndarray) -> None:
        self.t_ = np.array(value, dtype=np.float64).reshape(3, 1).copy()

    @property
    def dist(self) -> np.ndarray:
        return self.dist_

    @dist.setter
    def dist(self, value: np.ndarray) -> None:
        self.dist_ = np.array(value, dtype=np.float64).reshape(5, 1).copy()

    def rvec(self) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(self.R_)
        return rvec

    def t_wc(self) -> np.ndarray:
        """``-R^{-1} t``"""
        return -np.linalg.inv(self.R_) @ self.t_

    def clone(self) -> "Camera":
        return Camera(self.K_, self.R_, self.t_, self.dist_)

    # --- serialisation ----------------------------------------------------
    def to_vector(self) -> np.ndarray:
        """``Camera::ToVector``: [fx, fy, cx, cy, rvec(3), t(3), k1, k2, k3, p1, p2]"""
        v = np.empty(15, dtype=np.float64)
        v[0] = self.K_[0, 0]
        v[1] = self.K_[1, 1]
        v[2] = self.K_[0, 2]
        v[3] = self.K_[1, 2]
        rvec, _ = cv2.Rodrigues(self.R_)
        v[4:7] = rvec.reshape(3)
        v[7:10] = self.t_.reshape(3)
        v[10:15] = self.dist_.reshape(5)
        return v

    def from_vector(self, v) -> None:
        """``Camera::FromVector``"""
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.size != 15:
            raise ValueError(f"Expected camera vector size: 15, actual size :{v.size}")
        self.K_ = np.array([[v[0], 0.0, v[2]], [0.0, v[1], v[3]], [0.0, 0.0, 1.0]])
        R, _ = cv2.Rodrigues(v[4:7].reshape(3, 1))
        self.R_ = R
        self.t_ = v[7:10].reshape(3, 1).copy()
        self.dist_ = v[10:15].reshape(5, 1).copy()

    @staticmethod
    def from_vector_new(v) -> "Camera":
        cam = Camera()
        cam.from_vector(v)
        return cam

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Camera(f={self.K_[0, 0]:.3f}, c=({self.K_[0, 2]:.1f},{self.K_[1, 2]:.1f}))"


def pt3d_to_pix(camera: Camera, pt3d: np.ndarray) -> Optional[np.ndarray]:
    """``pt3d_to_pix``: returns ``None`` for the empty ``cv::Mat`` case."""
    pt_cam = camera.R @ np.asarray(pt3d, dtype=np.float64).reshape(3, 1) + camera.t
    near_plane = 1.0  # 1m
    if pt_cam[2, 0] < near_plane:
        return None
    pt_cam = pt_cam / pt_cam[2, 0]
    return camera.K @ pt_cam


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    return rvec.reshape(3)


def rodrigues_batch(rvecs: np.ndarray) -> np.ndarray:
    """Vectorised Rodrigues for a stack of rotation vectors.

    ``rvecs``: ``(N, 3)`` -> ``(N, 3, 3)``. Matches ``cv::Rodrigues`` including the
    small-angle branch (OpenCV switches to the first-order expansion when
    ``theta < DBL_EPSILON``).
    """
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(rvecs, axis=1)
    n = rvecs.shape[0]
    R = np.empty((n, 3, 3), dtype=np.float64)

    small = theta < np.finfo(np.float64).eps
    # general case: Rodrigues' formula
    safe_theta = np.where(small, 1.0, theta)
    u = rvecs / safe_theta[:, None]
    c = np.cos(theta)
    s = np.sin(theta)
    one_c = 1.0 - c

    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    R[:, 0, 0] = c + ux * ux * one_c
    R[:, 0, 1] = ux * uy * one_c - uz * s
    R[:, 0, 2] = ux * uz * one_c + uy * s
    R[:, 1, 0] = uy * ux * one_c + uz * s
    R[:, 1, 1] = c + uy * uy * one_c
    R[:, 1, 2] = uy * uz * one_c - ux * s
    R[:, 2, 0] = uz * ux * one_c - uy * s
    R[:, 2, 1] = uz * uy * one_c + ux * s
    R[:, 2, 2] = c + uz * uz * one_c

    if np.any(small):
        idx = np.nonzero(small)[0]
        rx, ry, rz = rvecs[idx, 0], rvecs[idx, 1], rvecs[idx, 2]
        R[idx] = np.stack(
            [
                np.stack([np.ones_like(rx), -rz, ry], axis=-1),
                np.stack([rz, np.ones_like(rx), -rx], axis=-1),
                np.stack([-ry, rx, np.ones_like(rx)], axis=-1),
            ],
            axis=1,
        )

    return R
