"""Port of ``src/core/krt_optimizer.{h,cc}``.

Single-camera ``K``/``R`` refinement used both for image registration inside
PTZ-IBA and for online relocalisation.  The residual functors are vectorised
over a batch of parameter vectors so that the Ceres-style central differences
stay cheap.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import lm
from .logging_utils import PLOGD
from .types import Camera, rodrigues_batch


class FactorType(IntEnum):
    """``KRTOptimizer::FACTOR_TYPE``"""

    F = 0
    FDist = 1
    Fxfy = 2
    FxfyDist = 3


# ``SubsetParameterization`` constant indices -> free indices of the 15-vector
_CONSTANT_INDICES = {
    FactorType.F: [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14],
    FactorType.Fxfy: [2, 3, 7, 8, 9, 10, 11, 12, 13, 14],
    FactorType.FDist: [1, 2, 3, 7, 8, 9, 11, 12, 13, 14],
    FactorType.FxfyDist: [2, 3, 7, 8, 9, 11, 12, 13, 14],
}


def _free_indices(factor_type: FactorType) -> np.ndarray:
    const = set(_CONSTANT_INDICES[factor_type])
    return np.array([i for i in range(15) if i not in const], dtype=np.int64)


# --------------------------------------------------------------------------
# Factors
# --------------------------------------------------------------------------
def factor_2d2d(params: np.ndarray, ray1: np.ndarray, uv2: np.ndarray, fxfy: bool) -> np.ndarray:
    """``Factor2d2d`` / ``Factor2d2dFxfy``: ``x2 = K2 R2 ray1``.

    ``params`` (E, 15), ``ray1`` (M, 3), ``uv2`` (M, 2) -> (E, M, 2)
    """
    R = rodrigues_batch(params[:, 4:7])
    p = np.einsum("eij,mj->emi", R, ray1)
    fx = params[:, 0][:, None]
    fy = (params[:, 1] if fxfy else params[:, 0])[:, None]
    cx = params[:, 2][:, None]
    cy = params[:, 3][:, None]

    z = p[:, :, 2]
    x_proj = fx * p[:, :, 0] / z + cx
    y_proj = fy * p[:, :, 1] / z + cy
    return np.stack([uv2[None, :, 0] - x_proj, uv2[None, :, 1] - y_proj], axis=-1)


def _distort(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distortion as spelled out in the C++ factors (dist = [k1,k2,k3,p1,p2])."""
    k1 = params[:, 10][:, None]
    k2 = params[:, 11][:, None]
    k3 = params[:, 12][:, None]
    p1 = params[:, 13][:, None]
    p2 = params[:, 14][:, None]

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r2 * r2 * r2
    xy = x * y
    x2 = x * x
    y2 = y * y
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    x_distorted = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
    y_distorted = y * radial + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
    return x_distorted, y_distorted


def factor_2d2d_dist(
    params: np.ndarray, ray1: np.ndarray, uv2: np.ndarray, valid: np.ndarray, fxfy: bool
) -> np.ndarray:
    """``Factor2d2dDist`` / ``Factor2d2dFxfyDist``.

    ``ray1`` already comes from the undistorted reference pixel; ``valid`` masks
    out the near-border pixels (residual forced to 0 by the C++ factor).
    """
    R = rodrigues_batch(params[:, 4:7])
    p = np.einsum("eij,mj->emi", R, ray1)
    z = p[:, :, 2]
    x = p[:, :, 0] / z
    y = p[:, :, 1] / z

    x_distorted, y_distorted = _distort(x, y, params)

    fx = params[:, 0][:, None]
    fy = (params[:, 1] if fxfy else params[:, 0])[:, None]
    cx = params[:, 2][:, None]
    cy = params[:, 3][:, None]
    x_proj = fx * x_distorted + cx
    y_proj = fy * y_distorted + cy

    res = np.stack([uv2[None, :, 0] - x_proj, uv2[None, :, 1] - y_proj], axis=-1)
    res[:, ~valid, :] = 0.0
    return res


def factor_2d3d(params: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray, fxfy: bool) -> np.ndarray:
    """``Factor2d3dDist`` / ``Factor2d3dFxfyDist`` (uses ``cv::projectPoints``)."""
    n_eval = params.shape[0]
    out = np.empty((n_eval, pts3d.shape[0], 2), dtype=np.float64)
    for e in range(n_eval):
        param = params[e].copy()
        if not fxfy:
            param[1] = param[0]
        cam = Camera.from_vector_new(param)
        proj, _ = cv2.projectPoints(pts3d, cam.rvec(), cam.t, cam.K, cam.dist)
        out[e] = pts2d - proj.reshape(-1, 2)
    return out


# --------------------------------------------------------------------------
# KRTOptimizer
# --------------------------------------------------------------------------
class KRTOptimizer:
    """``class KRTOptimizer``"""

    def __init__(self, max_iter: int, max_reproj_error: float, factor_type: FactorType):
        self.max_iter_ = max_iter
        self.max_reproj_error_ = max_reproj_error
        self.factor_type_ = FactorType(factor_type)

        self.num_iter_ = 0
        self.set_fixed_focal_ = False

        self.cam_curr_world_ = Camera()
        self.cam_curr_local_ = Camera()
        self.cam_curr_local_param_: np.ndarray = np.zeros(15)

        self.R_local_world_: Optional[np.ndarray] = None
        self.t_local_world_: Optional[np.ndarray] = None

        self._free_idx: np.ndarray = np.arange(15, dtype=np.int64)
        self._has_2d2d = False

        # residual block payloads
        self._ray1: Optional[np.ndarray] = None
        self._uv2: Optional[np.ndarray] = None
        self._valid: Optional[np.ndarray] = None
        self._pts2d: Optional[np.ndarray] = None
        self._pts3d_local: Optional[np.ndarray] = None

    # -- setup -------------------------------------------------------------
    def set_init_params(self, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: np.ndarray) -> None:
        self.cam_curr_world_.K = K
        self.cam_curr_world_.dist = dist
        self.cam_curr_world_.R = R
        self.cam_curr_world_.t = t

    def set_fixed_focal(self) -> None:
        self.set_fixed_focal_ = True

    def add_2d2d_constraints(
        self, cam_ref: Camera, kpts_ref: np.ndarray, kpts_curr: np.ndarray, matches: np.ndarray
    ) -> None:
        """``KRTOptimizer::Add2d2dConstraints``"""
        # reference frame becomes the local coordinate system
        self.R_local_world_ = cam_ref.R.copy()
        self.t_local_world_ = cam_ref.t.copy()

        cam_ref_local = Camera()
        cam_ref_local.K = cam_ref.K
        cam_ref_local.dist = cam_ref.dist
        cam_ref_local.R = np.eye(3)
        cam_ref_local.t = np.zeros((3, 1))

        # T_curr_local = T_curr_world * T_local_world^{-1}
        R_lw_inv = np.linalg.inv(self.R_local_world_)
        self.cam_curr_local_.K = self.cam_curr_world_.K
        self.cam_curr_local_.dist = self.cam_curr_world_.dist
        self.cam_curr_local_.R = self.cam_curr_world_.R @ R_lw_inv
        self.cam_curr_local_.t = (
            -self.cam_curr_world_.R @ R_lw_inv @ self.t_local_world_ + self.cam_curr_world_.t
        )

        self.cam_curr_local_param_ = self.cam_curr_local_.to_vector()

        uv1 = kpts_ref[matches[:, 0]].astype(np.float64)
        self._uv2 = kpts_curr[matches[:, 1]].astype(np.float64)

        K1 = cam_ref_local.K
        K1_inv = np.linalg.inv(K1)

        if self.factor_type_ in (FactorType.FDist, FactorType.FxfyDist):
            undist = cv2.undistortPoints(
                uv1.reshape(-1, 1, 2).astype(np.float64), K1, cam_ref_local.dist, None, K1
            ).reshape(-1, 2)
            width1 = K1[0, 2] * 2
            height1 = K1[1, 2] * 2
            self._valid = (
                (undist[:, 0] >= 0) & (undist[:, 0] < width1) & (undist[:, 1] >= 0) & (undist[:, 1] < height1)
            )
            pts = np.concatenate([undist, np.ones((undist.shape[0], 1))], axis=1)
            ray1 = pts @ K1_inv.T
            self._ray1 = ray1 / np.linalg.norm(ray1, axis=1, keepdims=True)
        else:
            pts = np.concatenate([uv1, np.ones((uv1.shape[0], 1))], axis=1)
            ray1 = pts @ K1_inv.T
            if self.factor_type_ == FactorType.F:
                # Factor2d2d normalises the ray; Factor2d2dFxfy does not
                ray1 = ray1 / np.linalg.norm(ray1, axis=1, keepdims=True)
            self._ray1 = ray1
            self._valid = np.ones(ray1.shape[0], dtype=bool)

        self._has_2d2d = True
        self._free_idx = _free_indices(self.factor_type_)

    def add_2d3d_constraints(self, pts2d: np.ndarray, pts3d: np.ndarray) -> None:
        """``KRTOptimizer::Add2d3dConstraints``"""
        pts2d = np.asarray(pts2d, dtype=np.float64).reshape(-1, 2)
        pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
        if pts2d.shape[0] != pts3d.shape[0] or pts2d.shape[0] == 0:
            return
        self._pts2d = pts2d
        self._pts3d_local = (self.R_local_world_ @ pts3d.T + self.t_local_world_).T

    # -- residuals ---------------------------------------------------------
    def _residuals(self, params_free: np.ndarray) -> np.ndarray:
        n_eval = params_free.shape[0]
        params = np.repeat(self.cam_curr_local_param_[None, :], n_eval, axis=0)
        params[:, self._free_idx] = params_free

        blocks: List[np.ndarray] = []
        if self._has_2d2d:
            fxfy = self.factor_type_ in (FactorType.Fxfy, FactorType.FxfyDist)
            if self.factor_type_ in (FactorType.FDist, FactorType.FxfyDist):
                res = factor_2d2d_dist(params, self._ray1, self._uv2, self._valid, fxfy)
            else:
                res = factor_2d2d(params, self._ray1, self._uv2, fxfy)
            blocks.append(res.reshape(n_eval, -1))
        if self._pts2d is not None:
            fxfy = self.factor_type_ in (FactorType.Fxfy, FactorType.FxfyDist)
            res = factor_2d3d(params, self._pts3d_local, self._pts2d, fxfy)
            blocks.append(res.reshape(n_eval, -1))

        if not blocks:
            return np.zeros((n_eval, 0))
        return np.concatenate(blocks, axis=1)

    # -- solve -------------------------------------------------------------
    def solve(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """``KRTOptimizer::Solve``: returns ``(ok, K, R, t, dist)``."""
        options = lm.SolverOptions(max_num_iterations=self.max_iter_, minimizer_progress_to_stdout=False)
        system = lm.DenseNumericSystem(self._residuals)
        x0 = self.cam_curr_local_param_[self._free_idx].copy()

        x, summary = lm.minimize(system, x0, options)
        self.cam_curr_local_param_[self._free_idx] = x
        self.num_iter_ = summary.num_successful_steps

        if not self._check_results(summary):
            return False, None, None, None, None

        return (True,) + self._obtain_refined_camera_params()

    def _check_results(self, summary: lm.SolverSummary) -> bool:
        """``KRTOptimizer::CheckResults``"""
        if summary.num_residuals == 0:
            return False
        init_reproj_error = math.sqrt(2) * math.sqrt((2 * summary.initial_cost) / summary.num_residuals)
        final_reproj_error = math.sqrt(2) * math.sqrt((2 * summary.final_cost) / summary.num_residuals)

        PLOGD(summary.brief_report())
        PLOGD(
            f"Init reprojection error: {init_reproj_error}, final reprojection error: {final_reproj_error}"
        )

        if summary.termination_type != lm.CONVERGENCE:
            return False

        if final_reproj_error >= self.max_reproj_error_:
            return False

        cam = Camera.from_vector_new(self.cam_curr_local_param_)
        fx, fy = cam.K[0, 0], cam.K[1, 1]
        cx, cy = cam.K[0, 2], cam.K[1, 2]
        fov_x = math.atan(cx / fx) * 2 * 180 / math.pi
        fov_y = math.atan(cy / fy) * 2 * 180 / math.pi
        if fov_x < 0 or fov_x > 170 or fov_y < 0 or fov_y > 170:
            PLOGD(f"FOV invalid! fov_x: {fov_x}, fov_y: {fov_y}")
            return False

        return True

    def _obtain_refined_camera_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``KRTOptimizer::ObtainRefinedCameraParams``"""
        param = self.cam_curr_local_param_.copy()
        if self.factor_type_ in (FactorType.F, FactorType.FDist):
            param[1] = param[0]  # fx = fy

        self.cam_curr_local_.from_vector(param)

        # T_curr_world = T_curr_local * T_local_world
        self.cam_curr_world_.K = self.cam_curr_local_.K
        self.cam_curr_world_.dist = self.cam_curr_local_.dist
        self.cam_curr_world_.R = self.cam_curr_local_.R @ self.R_local_world_
        self.cam_curr_world_.t = self.cam_curr_local_.R @ self.t_local_world_ + self.cam_curr_local_.t

        return (
            self.cam_curr_world_.K,
            self.cam_curr_world_.R,
            self.cam_curr_world_.t,
            self.cam_curr_world_.dist,
        )

    # -- diagnostics -------------------------------------------------------
    def cal_2d2d_reproj_error(
        self, cam_ref: Camera, kpts_ref: np.ndarray, kpts_curr: np.ndarray, matches: np.ndarray
    ) -> float:
        """``KRTOptimizer::Cal2d2dReprojError``"""
        K1 = cam_ref.K
        K1_inv = np.linalg.inv(K1)
        uv1 = kpts_ref[matches[:, 0]].astype(np.float64)
        uv2 = kpts_curr[matches[:, 1]].astype(np.float64)
        fxfy = self.factor_type_ in (FactorType.Fxfy, FactorType.FxfyDist)
        params = self.cam_curr_local_param_[None, :]

        if self.factor_type_ in (FactorType.FDist, FactorType.FxfyDist):
            undist = cv2.undistortPoints(
                uv1.reshape(-1, 1, 2).astype(np.float64), K1, cam_ref.dist, None, K1
            ).reshape(-1, 2)
            width1 = K1[0, 2] * 2
            height1 = K1[1, 2] * 2
            valid = (
                (undist[:, 0] >= 0) & (undist[:, 0] < width1) & (undist[:, 1] >= 0) & (undist[:, 1] < height1)
            )
            pts = np.concatenate([undist, np.ones((undist.shape[0], 1))], axis=1)
            ray1 = pts @ K1_inv.T
            ray1 = ray1 / np.linalg.norm(ray1, axis=1, keepdims=True)
            res = factor_2d2d_dist(params, ray1, uv2, valid, fxfy)[0]
        else:
            pts = np.concatenate([uv1, np.ones((uv1.shape[0], 1))], axis=1)
            ray1 = pts @ K1_inv.T
            if self.factor_type_ == FactorType.F:
                ray1 = ray1 / np.linalg.norm(ray1, axis=1, keepdims=True)
            res = factor_2d2d(params, ray1, uv2, fxfy)[0]

        num_observation = float(matches.shape[0])
        return float(math.sqrt(np.sum(res * res) / num_observation))

    def cal_2d3d_reproj_error(self, pts2d: np.ndarray, pts3d: np.ndarray) -> float:
        """``KRTOptimizer::Cal2d3dReprojError``"""
        pts2d = np.asarray(pts2d, dtype=np.float64).reshape(-1, 2)
        pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
        if pts2d.shape[0] != pts3d.shape[0] or pts2d.shape[0] == 0:
            return -1.0
        pts3d_local = (self.R_local_world_ @ pts3d.T + self.t_local_world_).T
        fxfy = self.factor_type_ in (FactorType.Fxfy, FactorType.FxfyDist)
        res = factor_2d3d(self.cam_curr_local_param_[None, :], pts3d_local, pts2d, fxfy)[0]
        return float(math.sqrt(np.sum(res * res) / pts2d.shape[0]))
