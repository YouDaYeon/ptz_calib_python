"""Port of ``src/core/ptzray_optimizer.{h,cc}``.

Global bundle adjustment over ray landmarks (paper eq. 7) plus the optional
2D-3D georeferencing term (eq. 8).  Parameter blocks follow the C++ code:

* ``intrinsics`` (9) ``[fx, fy, cx, cy, k1, k2, k3, p1, p2]`` per ``shared_ic_id``
  (the default mapping is one block *per image*, i.e. no sharing)
* ``extrinsics`` (6) ``[rvec, t]`` per image, ``t`` held constant
* ``rays``       (3) per track
* ``disp``       (3) only for ``PTZRayDistDisp``
* ``tlw``        (6) only when 2D-3D constraints exist
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from . import lm
from .logging_utils import PLOGI, PLOGW
from .tracks import Tracks, TracksBuilder, find_max_covisible, length
from .types import Camera, ImageFeatures, MatchesInfo, Ray, rodrigues_batch

KPENALTY = 1000000.0


class FactorType(IntEnum):
    """``enum FACTOR_TYPE``"""

    PTZRay = 0
    PTZRayDist = 1
    PTZRayFxfyDist = 2
    PTZRayDistDisp = 3


# ``SubsetParameterization`` on the 9-dim intrinsics block
_FREE_INTRINSICS = {
    FactorType.PTZRay: np.array([0, 1], dtype=np.int64),  # const: cx, cy, k1, k2, k3, p1, p2
    FactorType.PTZRayDist: np.array([0, 1, 4], dtype=np.int64),  # const: cx, cy, k2, k3, p1, p2
    FactorType.PTZRayFxfyDist: np.array([0, 1, 4], dtype=np.int64),
    FactorType.PTZRayDistDisp: np.array([0, 1, 4], dtype=np.int64),
}


def t_l_w_from_param(tlw: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """``PTZRayOptimizer::T_l_w``"""
    R_l_w, _ = cv2.Rodrigues(np.asarray(tlw[0:3], dtype=np.float64).reshape(3, 1))
    t_l_w = np.asarray(tlw[3:6], dtype=np.float64).reshape(3, 1)
    return R_l_w, t_l_w


def _distort(x: np.ndarray, y: np.ndarray, k1, k2, k3, p1, p2) -> Tuple[np.ndarray, np.ndarray]:
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


class PTZRayOptimizer(lm.SchurSystem):
    """``class PTZRayOptimizer``"""

    def __init__(
        self,
        features: Sequence[ImageFeatures],
        matches_info: Sequence[MatchesInfo],
        cameras: Sequence[Camera],
        cam_ids: Optional[Set[int]],
        max_iter: int,
        factor_type: FactorType,
        pixels: Optional[Sequence[np.ndarray]] = None,
        pts3d: Optional[Sequence[np.ndarray]] = None,
    ):
        self.features_ = list(features)
        self.matches_info_ = list(matches_info)
        self.cameras_ = [c.clone() for c in cameras]
        self.pixels_ = list(pixels) if pixels is not None else []
        self.pts3d_ = list(pts3d) if pts3d is not None else []
        self.num_cams_ = len(cameras)
        self.max_iter_ = max_iter
        self.type_ = FactorType(factor_type)

        if not cam_ids:
            self.cam_ids_ = set(range(self.num_cams_))
        else:
            self.cam_ids_ = set(int(i) for i in cam_ids)

        self.shared_ic_ids_ = list(range(self.num_cams_))

        self.T_l_w_: Optional[np.ndarray] = None
        self.tlw_param_ = np.zeros(6, dtype=np.float64)
        self.disp_param_ = np.zeros(3, dtype=np.float64)
        self.intrinsics_param_: Dict[int, np.ndarray] = {}
        self.extrinsics_param_: Dict[int, np.ndarray] = {}
        self.rays_param_: Dict[int, np.ndarray] = {}

        self.tracks_: Tracks = {}
        self.track_len_ = 0
        self.max_track_len_ = 0
        self.min_track_len_ = 0

        self.summary_: Optional[lm.SolverSummary] = None
        self.init_reproj_error_all_ = 0.0
        self.final_reproj_error_all_ = 0.0
        self.final_reproj_error_2d2d_ = 0.0
        self.final_reproj_error_2d3d_ = 0.0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def set_shared_intrinsics(self, shared_ic_ids: Sequence[int]) -> None:
        """``PTZRayOptimizer::SetSharedIntrinsics``"""
        if len(shared_ic_ids) != len(self.cameras_):
            PLOGW(
                f"Set shared intrinsics failed, length not matched: {len(self.cameras_)} - {len(shared_ic_ids)}"
            )
            return
        self.shared_ic_ids_ = list(shared_ic_ids)

    def final_reproj_error_all(self) -> float:
        return self.final_reproj_error_all_

    def final_reproj_error_2d2d(self) -> float:
        return self.final_reproj_error_2d2d_

    def final_reproj_error_2d3d(self) -> float:
        return self.final_reproj_error_2d3d_

    def solve(self, cameras: List[Camera]) -> Tuple[bool, List[List[Ray]]]:
        """``PTZRayOptimizer::Solve``: updates ``cameras`` in place."""
        rays: List[List[Ray]] = []
        if not self._check_valid():
            return False, rays

        self._find_tracks()
        self._set_init_trans_local_to_world()
        self._set_up_initial_camera_params()
        self._build_problem()

        if self._num_obs_a == 0 and self._num_obs_b == 0:
            return False, rays

        options = lm.SolverOptions(max_num_iterations=self.max_iter_)
        x0 = self._pack_params()
        lm.SchurSystem.__init__(self, self.n_cam, self.n_ray)
        x, summary = lm.minimize(self, x0, options)
        self._unpack_params(x)
        self.summary_ = summary

        self._cal_reproj_error()

        PLOGI(summary.brief_report())
        PLOGI(
            f"Init reprojection error all: {self.init_reproj_error_all_}, "
            f"final reprojection error all: {self.final_reproj_error_all_}"
        )

        if summary.termination_type == lm.CONVERGENCE:
            rays = self._obtain_refined_camera_params(cameras)
            return True, rays
        return False, rays

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _is_candidate(self, image_id: int) -> bool:
        return image_id in self.cam_ids_

    def _check_valid(self) -> bool:
        """``PTZRayOptimizer::CheckValid``"""
        if self.num_cams_ == 0:
            return False
        if len(self.features_) != self.num_cams_:
            return False
        if self.max_iter_ <= 0:
            return False
        if self.pixels_:
            if len(self.pixels_) != self.num_cams_ or len(self.pts3d_) != self.num_cams_:
                return False
            for i in range(self.num_cams_):
                if len(self.pixels_[i]) != len(self.pts3d_[i]):
                    return False
        return True

    def _find_tracks(self) -> None:
        """``PTZRayOptimizer::FindTracks``"""
        builder = TracksBuilder()
        builder.build(self.matches_info_)
        builder.filter(4)
        self.tracks_ = builder.export_to_stl()

        self.track_len_, self.max_track_len_, self.min_track_len_ = length(self.tracks_)
        mean_track_length = self.track_len_ / float(len(self.tracks_)) if self.tracks_ else float("nan")
        PLOGI(
            f"Tracks number: {len(self.tracks_)}, total track length: {self.track_len_}, "
            f"mean track length: {mean_track_length}, min track length: {self.min_track_len_}, "
            f"max track length: {self.max_track_len_}"
        )

        max_connect_imgs = find_max_covisible(self.tracks_, self.num_cams_)
        PLOGI(f"Max co-visible number: {len(max_connect_imgs)}, total number: {self.num_cams_}")

    def _set_init_trans_local_to_world(self) -> bool:
        """``PTZRayOptimizer::SetInitTransLocalToWorld``"""
        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            if not self.pixels_ or len(self.pixels_[i]) == 0:
                continue

            obj = np.asarray(self.pts3d_[i], dtype=np.float64).reshape(-1, 3)
            img = np.asarray(self.pixels_[i], dtype=np.float64).reshape(-1, 2)
            try:
                pnp_success, rvec, tvec = cv2.solvePnP(
                    obj, img, self.cameras_[i].K, self.cameras_[i].dist, flags=cv2.SOLVEPNP_EPNP
                )
            except cv2.error:
                pnp_success = False
            if not pnp_success:
                PLOGW("SolvePnP failure")
                continue

            R, _ = cv2.Rodrigues(rvec)

            p3d = R @ obj[0].reshape(3, 1) + tvec
            if p3d[2, 0] < 0 or np.linalg.det(R) < 0.0:
                PLOGW("SolvePnP failure")
                continue

            pts3d_float = obj.astype(np.float32)
            predict, _ = cv2.projectPoints(pts3d_float, rvec, tvec, self.cameras_[i].K, None)
            predict = predict.reshape(-1, 2)
            reproj_error = math.sqrt(float(np.sum((predict - img) ** 2)) / predict.shape[0])
            if reproj_error > 300:
                PLOGW(f"Init reprojection error too large: {reproj_error}")
                continue
            PLOGI(f"Init reprojection error: {reproj_error}")

            T_i_w = np.eye(4)
            T_i_w[0:3, 0:3] = R
            T_i_w[0:3, 3:4] = tvec

            T_i_l = np.eye(4)
            T_i_l[0:3, 0:3] = self.cameras_[i].R
            T_i_l[0:3, 3:4] = self.cameras_[i].t

            self.T_l_w_ = np.linalg.inv(T_i_l) @ T_i_w
            R_l_w = self.T_l_w_[0:3, 0:3]
            t_l_w = self.T_l_w_[0:3, 3]
            rvec_l_w, _ = cv2.Rodrigues(R_l_w)

            self.tlw_param_ = np.array(
                [rvec_l_w[0, 0], rvec_l_w[1, 0], rvec_l_w[2, 0], t_l_w[0], t_l_w[1], t_l_w[2]],
                dtype=np.float64,
            )
            return True

        self.tlw_param_ = np.zeros(6, dtype=np.float64)
        return False

    def _pix2ray(self, track: Dict[int, int]) -> np.ndarray:
        """``PTZRayOptimizer::Pix2Ray``: mean of the normalised observation rays."""
        ray = np.zeros(3, dtype=np.float64)
        track_size = 0
        for image_id, feature_id in track.items():
            if not self._is_candidate(image_id):
                continue
            pt = self.features_[image_id].keypoints[feature_id]
            uv = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
            cam = self.cameras_[image_id]
            ray_temp = np.linalg.inv(cam.R) @ np.linalg.inv(cam.K) @ uv
            ray_temp = ray_temp / np.linalg.norm(ray_temp)
            ray += ray_temp
            track_size += 1

        with np.errstate(invalid="ignore", divide="ignore"):
            ray = ray / track_size
            ray = ray / np.linalg.norm(ray)
        return ray

    def _set_up_initial_camera_params(self) -> None:
        """``PTZRayOptimizer::SetUpInitialCameraParams``"""
        self.intrinsics_param_ = {}
        self.extrinsics_param_ = {}

        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            param = self.cameras_[i].to_vector()

            ic_id = self.shared_ic_ids_[i]
            if ic_id not in self.intrinsics_param_:
                self.intrinsics_param_[ic_id] = np.array(
                    [param[0], param[1], param[2], param[3], param[10], param[11], param[12], param[13], param[14]],
                    dtype=np.float64,
                )

            self.extrinsics_param_[i] = np.array(
                [param[4], param[5], param[6], param[7], param[8], param[9]], dtype=np.float64
            )

        self.disp_param_ = np.zeros(3, dtype=np.float64)

        # triangulate rays
        self.rays_param_ = {}
        for track_id, track in self.tracks_.items():
            ray = self._pix2ray(track)
            self.rays_param_[track_id] = ray

    # ------------------------------------------------------------------
    # problem construction (AddConstraints2d2d / AddConstraints2d3d)
    # ------------------------------------------------------------------
    def _build_problem(self) -> None:
        self._free_intr = _FREE_INTRINSICS[self.type_]
        self.n_fi = int(self._free_intr.size)

        self._img_ids = sorted(self.cam_ids_)
        self._img_local = {img: k for k, img in enumerate(self._img_ids)}
        self.n_img = len(self._img_ids)

        self._ic_ids = sorted({self.shared_ic_ids_[i] for i in self._img_ids})
        self._ic_local = {ic: k for k, ic in enumerate(self._ic_ids)}
        self.n_ic = len(self._ic_ids)

        # --- 2d-2d observations -------------------------------------------
        ic_idx: List[int] = []
        ext_idx: List[int] = []
        ray_idx: List[int] = []
        uv_list: List[Tuple[float, float]] = []
        weights: List[float] = []

        self._ray_track_ids: List[int] = []
        track_to_ray: Dict[int, int] = {}

        for track_id, track in self.tracks_.items():
            weight = float(len(track))
            for image_id, feature_id in track.items():
                if not self._is_candidate(image_id):
                    continue
                if track_id not in track_to_ray:
                    track_to_ray[track_id] = len(self._ray_track_ids)
                    self._ray_track_ids.append(track_id)
                pt = self.features_[image_id].keypoints[feature_id]
                ic_idx.append(self._ic_local[self.shared_ic_ids_[image_id]])
                ext_idx.append(self._img_local[image_id])
                ray_idx.append(track_to_ray[track_id])
                uv_list.append((float(pt[0]), float(pt[1])))
                weights.append(weight)

        self._track_to_ray = track_to_ray
        self._ic_idx = np.asarray(ic_idx, dtype=np.int64)
        self._ext_idx = np.asarray(ext_idx, dtype=np.int64)
        self._obs_ray = np.asarray(ray_idx, dtype=np.int64)
        self._uv = np.asarray(uv_list, dtype=np.float64).reshape(-1, 2)
        self._w = np.sqrt(np.asarray(weights, dtype=np.float64)).reshape(-1, 1)
        self._num_obs_a = self._ic_idx.size
        self.n_ray = len(self._ray_track_ids)

        # --- 2d-3d observations -------------------------------------------
        ic_idx3: List[int] = []
        ext_idx3: List[int] = []
        uv3: List[Tuple[float, float]] = []
        pts3: List[Tuple[float, float, float]] = []
        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            if not self.pixels_ or len(self.pixels_[i]) == 0:
                continue
            pix = np.asarray(self.pixels_[i], dtype=np.float64).reshape(-1, 2)
            p3d = np.asarray(self.pts3d_[i], dtype=np.float64).reshape(-1, 3)
            for j in range(pix.shape[0]):
                ic_idx3.append(self._ic_local[self.shared_ic_ids_[i]])
                ext_idx3.append(self._img_local[i])
                uv3.append((pix[j, 0], pix[j, 1]))
                pts3.append((p3d[j, 0], p3d[j, 1], p3d[j, 2]))

        self._ic_idx3 = np.asarray(ic_idx3, dtype=np.int64)
        self._ext_idx3 = np.asarray(ext_idx3, dtype=np.int64)
        self._uv3 = np.asarray(uv3, dtype=np.float64).reshape(-1, 2)
        self._pts3 = np.asarray(pts3, dtype=np.float64).reshape(-1, 3)
        self._num_obs_b = self._ic_idx3.size

        self._use_tlw = self._num_obs_b > 0
        self._use_disp = self.type_ == FactorType.PTZRayDistDisp and self._num_obs_a > 0

        # --- parameter vector layout --------------------------------------
        self._intr_off = 0
        self._ext_off = self._intr_off + self.n_ic * self.n_fi
        self._disp_off = self._ext_off + self.n_img * 3
        self._tlw_off = self._disp_off + (3 if self._use_disp else 0)
        self.n_cam = self._tlw_off + (6 if self._use_tlw else 0)

        # per-observation camera column indices
        base_ic = self._intr_off + self._ic_idx[:, None] * self.n_fi + np.arange(self.n_fi)[None, :]
        base_ext = self._ext_off + self._ext_idx[:, None] * 3 + np.arange(3)[None, :]
        cols_a = [base_ic, base_ext]
        if self._use_disp:
            cols_a.append(np.repeat((self._disp_off + np.arange(3))[None, :], self._num_obs_a, axis=0))
        self._cam_cols_a = np.concatenate(cols_a, axis=1) if self._num_obs_a else np.zeros((0, 0), dtype=np.int64)

        if self._num_obs_b:
            base_ic3 = self._intr_off + self._ic_idx3[:, None] * self.n_fi + np.arange(self.n_fi)[None, :]
            base_ext3 = self._ext_off + self._ext_idx3[:, None] * 3 + np.arange(3)[None, :]
            cols_b = [base_ic3, base_ext3]
            if self._use_disp:
                cols_b.append(np.repeat((self._disp_off + np.arange(3))[None, :], self._num_obs_b, axis=0))
            cols_b.append(np.repeat((self._tlw_off + np.arange(6))[None, :], self._num_obs_b, axis=0))
            self._cam_cols_b = np.concatenate(cols_b, axis=1)
        else:
            self._cam_cols_b = np.zeros((0, 0), dtype=np.int64)

        # constant parts of the parameter blocks
        self._intr0 = np.stack([self.intrinsics_param_[ic] for ic in self._ic_ids]) if self.n_ic else np.zeros((0, 9))
        self._ext0 = np.stack([self.extrinsics_param_[i] for i in self._img_ids]) if self.n_img else np.zeros((0, 6))
        self._rays0 = (
            np.stack([self.rays_param_[t] for t in self._ray_track_ids])
            if self.n_ray
            else np.zeros((0, 3))
        )

    def _pack_params(self) -> np.ndarray:
        x = np.zeros(self.n_cam + 3 * self.n_ray, dtype=np.float64)
        x[self._intr_off: self._ext_off] = self._intr0[:, self._free_intr].reshape(-1)
        x[self._ext_off: self._ext_off + self.n_img * 3] = self._ext0[:, 0:3].reshape(-1)
        if self._use_disp:
            x[self._disp_off: self._disp_off + 3] = self.disp_param_
        if self._use_tlw:
            x[self._tlw_off: self._tlw_off + 6] = self.tlw_param_
        x[self.n_cam:] = self._rays0.reshape(-1)
        return x

    def _unpack_params(self, x: np.ndarray) -> None:
        intr = self._intr0.copy()
        intr[:, self._free_intr] = x[self._intr_off: self._ext_off].reshape(self.n_ic, self.n_fi)
        for k, ic in enumerate(self._ic_ids):
            self.intrinsics_param_[ic] = intr[k]

        ext = self._ext0.copy()
        ext[:, 0:3] = x[self._ext_off: self._ext_off + self.n_img * 3].reshape(self.n_img, 3)
        for k, img in enumerate(self._img_ids):
            self.extrinsics_param_[img] = ext[k]

        if self._use_disp:
            self.disp_param_ = x[self._disp_off: self._disp_off + 3].copy()
        if self._use_tlw:
            self.tlw_param_ = x[self._tlw_off: self._tlw_off + 6].copy()

        rays = x[self.n_cam:].reshape(self.n_ray, 3)
        for k, track_id in enumerate(self._ray_track_ids):
            self.rays_param_[track_id] = rays[k]

    # ------------------------------------------------------------------
    # residuals
    # ------------------------------------------------------------------
    def _expand(self, xs: np.ndarray):
        n_eval = xs.shape[0]
        intr = np.repeat(self._intr0[None, :, :], n_eval, axis=0)
        intr[:, :, self._free_intr] = xs[:, self._intr_off: self._ext_off].reshape(
            n_eval, self.n_ic, self.n_fi
        )
        ext = np.repeat(self._ext0[None, :, :], n_eval, axis=0)
        ext[:, :, 0:3] = xs[:, self._ext_off: self._ext_off + self.n_img * 3].reshape(n_eval, self.n_img, 3)

        disp = (
            xs[:, self._disp_off: self._disp_off + 3]
            if self._use_disp
            else np.repeat(self.disp_param_[None, :], n_eval, axis=0)
        )
        tlw = (
            xs[:, self._tlw_off: self._tlw_off + 6]
            if self._use_tlw
            else np.repeat(self.tlw_param_[None, :], n_eval, axis=0)
        )
        rays = xs[:, self.n_cam:].reshape(n_eval, self.n_ray, 3)
        return intr, ext, disp, tlw, rays

    def _residual_a(self, intr, ext, disp, rays, weighted: bool = True) -> np.ndarray:
        """2D-2D ray factors -> (E, N, 2)"""
        n_eval = intr.shape[0]
        R = rodrigues_batch(ext[:, :, 0:3].reshape(-1, 3)).reshape(n_eval, self.n_img, 3, 3)
        Ro = R[:, self._ext_idx]  # (E, N, 3, 3)
        v = rays[:, self._obs_ray]  # (E, N, 3)

        if self.type_ != FactorType.PTZRayDist:
            # every other factor normalises the ray first
            v = v / np.linalg.norm(v, axis=2, keepdims=True)

        p = np.einsum("enij,enj->eni", Ro, v)

        ic = intr[:, self._ic_idx]  # (E, N, 9)
        fx = ic[:, :, 0]
        fy = ic[:, :, 1] if self.type_ == FactorType.PTZRayFxfyDist else ic[:, :, 0]
        cx = ic[:, :, 2]
        cy = ic[:, :, 3]

        behind = None
        if self.type_ == FactorType.PTZRayDist:
            behind = p[:, :, 2] < 0

        z = p[:, :, 2]
        if self.type_ == FactorType.PTZRayDistDisp:
            displacement = disp[:, 0:1] + disp[:, 1:2] * fx + disp[:, 2:3] * fx * fx
            z = z + displacement

        with np.errstate(divide="ignore", invalid="ignore"):
            x = p[:, :, 0] / z
            y = p[:, :, 1] / z

        if self.type_ == FactorType.PTZRay:
            x_proj = fx * x + cx
            y_proj = fy * y + cy
        else:
            xd, yd = _distort(x, y, ic[:, :, 4], ic[:, :, 5], ic[:, :, 6], ic[:, :, 7], ic[:, :, 8])
            x_proj = fx * xd + cx
            y_proj = fy * yd + cy

        res = np.stack([self._uv[None, :, 0] - x_proj, self._uv[None, :, 1] - y_proj], axis=-1)

        if behind is not None and np.any(behind):
            res[behind] = KPENALTY

        if weighted:
            res = res * self._w[None, :, :]
        return res

    def _residual_b(self, intr, ext, disp, tlw) -> np.ndarray:
        """2D-3D reprojection factors -> (E, M, 2)"""
        n_eval = intr.shape[0]
        R = rodrigues_batch(ext[:, :, 0:3].reshape(-1, 3)).reshape(n_eval, self.n_img, 3, 3)
        Ro = R[:, self._ext_idx3]  # (E, M, 3, 3)

        R_l_w = rodrigues_batch(tlw[:, 0:3])  # (E, 3, 3)
        t_l_w = tlw[:, 3:6]
        pt3d_l = np.einsum("eij,mj->emi", R_l_w, self._pts3) + t_l_w[:, None, :]

        pt_cam = np.einsum("emij,emj->emi", Ro, pt3d_l)

        ic = intr[:, self._ic_idx3]
        fx = ic[:, :, 0]
        fy = ic[:, :, 1]  # Reproj2d3dFactor always uses intrinsics[1]
        cx = ic[:, :, 2]
        cy = ic[:, :, 3]

        z = pt_cam[:, :, 2]
        if self.type_ == FactorType.PTZRayDistDisp:
            displacement = disp[:, 0:1] + disp[:, 1:2] * fx + disp[:, 2:3] * fx * fx
            z = z + displacement

        with np.errstate(divide="ignore", invalid="ignore"):
            x = pt_cam[:, :, 0] / z
            y = pt_cam[:, :, 1] / z

        xd, yd = _distort(x, y, ic[:, :, 4], ic[:, :, 5], ic[:, :, 6], ic[:, :, 7], ic[:, :, 8])
        x_proj = fx * xd + cx
        y_proj = fy * yd + cy

        return np.stack([self._uv3[None, :, 0] - x_proj, self._uv3[None, :, 1] - y_proj], axis=-1)

    def _eval(self, xs: np.ndarray, need_a: bool = True, need_b: bool = True):
        intr, ext, disp, tlw, rays = self._expand(xs)
        ra = self._residual_a(intr, ext, disp, rays) if (need_a and self._num_obs_a) else None
        rb = self._residual_b(intr, ext, disp, tlw) if (need_b and self._num_obs_b) else None
        return ra, rb

    # -- lm.TrustRegionSystem ------------------------------------------
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        ra, rb = self._eval(x[None, :])
        parts = []
        if ra is not None:
            parts.append(ra[0].reshape(-1))
        if rb is not None:
            parts.append(rb[0].reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0)

    def linearize(self, x: np.ndarray, f: np.ndarray) -> None:
        na, mb = self._num_obs_a, self._num_obs_b
        nb_a = self._cam_cols_a.shape[1]
        nb_b = self._cam_cols_b.shape[1] if mb else 0
        jac_a = np.zeros((na, 2, nb_a)) if na else np.zeros((0, 2, 0))
        jac_r = np.zeros((na, 2, 3)) if na else np.zeros((0, 2, 3))
        jac_b = np.zeros((mb, 2, nb_b)) if mb else np.zeros((0, 2, 0))

        def diff(group: np.ndarray, owner_a, owner_b, need_a: bool, need_b: bool):
            """Central differences over a set of columns with disjoint residual support."""
            delta = lm.central_difference_step(x[group])
            xs = np.repeat(x[None, :], 2, axis=0)
            xs[0, group] += delta
            xs[1, group] -= delta
            ra, rb = self._eval(xs, need_a=need_a, need_b=need_b)
            out_a = None
            out_b = None
            if ra is not None:
                da = delta[owner_a] if owner_a is not None else delta[0]
                out_a = (ra[0] - ra[1]) / (2.0 * np.asarray(da).reshape(-1, 1))
            if rb is not None:
                db = delta[owner_b] if owner_b is not None else delta[0]
                out_b = (rb[0] - rb[1]) / (2.0 * np.asarray(db).reshape(-1, 1))
            return out_a, out_b

        # intrinsics: one group per free component (blocks are disjoint per ic id)
        for j in range(self.n_fi):
            group = self._intr_off + np.arange(self.n_ic) * self.n_fi + j
            da, db = diff(group, self._ic_idx if na else None, self._ic_idx3 if mb else None, na > 0, mb > 0)
            if da is not None:
                jac_a[:, :, j] = da
            if db is not None:
                jac_b[:, :, j] = db

        # extrinsics rvec: one group per component (images are disjoint)
        for j in range(3):
            group = self._ext_off + np.arange(self.n_img) * 3 + j
            da, db = diff(group, self._ext_idx if na else None, self._ext_idx3 if mb else None, na > 0, mb > 0)
            if da is not None:
                jac_a[:, :, self.n_fi + j] = da
            if db is not None:
                jac_b[:, :, self.n_fi + j] = db

        col = self.n_fi + 3
        if self._use_disp:
            for j in range(3):
                group = np.array([self._disp_off + j])
                da, db = diff(group, None, None, na > 0, mb > 0)
                if da is not None:
                    jac_a[:, :, col + j] = da
                if db is not None:
                    jac_b[:, :, col + j] = db
            col += 3

        if self._use_tlw:
            for j in range(6):
                group = np.array([self._tlw_off + j])
                _da, db = diff(group, None, None, False, True)
                if db is not None:
                    jac_b[:, :, col + j] = db

        # rays
        for j in range(3):
            if not self.n_ray:
                break
            group = self.n_cam + np.arange(self.n_ray) * 3 + j
            da, _db = diff(group, self._obs_ray, None, True, False)
            if da is not None:
                jac_r[:, :, j] = da

        self.set_blocks(self._cam_cols_a, jac_a, self._obs_ray, jac_r, self._cam_cols_b, jac_b, f)

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------
    def _cal_reproj_error(self) -> None:
        """``PTZRayOptimizer::CalReprojError``"""
        s = self.summary_
        if s is None or s.num_residuals == 0:
            return
        self.init_reproj_error_all_ = math.sqrt(2) * math.sqrt((2 * s.initial_cost) / s.num_residuals)
        self.final_reproj_error_all_ = math.sqrt(2) * math.sqrt((2 * s.final_cost) / s.num_residuals)

        x = self._pack_params_from_state()
        intr, ext, disp, tlw, rays = self._expand(x[None, :])

        if self._num_obs_a:
            res = self._residual_a(intr, ext, disp, rays, weighted=False)[0]
            self.final_reproj_error_2d2d_ = math.sqrt(float(np.sum(res * res)) / self._num_obs_a)
        if self._num_obs_b:
            res = self._residual_b(intr, ext, disp, tlw)[0]
            self.final_reproj_error_2d3d_ = math.sqrt(float(np.sum(res * res)) / self._num_obs_b)

    def _pack_params_from_state(self) -> np.ndarray:
        x = np.zeros(self.n_cam + 3 * self.n_ray, dtype=np.float64)
        intr = np.stack([self.intrinsics_param_[ic] for ic in self._ic_ids]) if self.n_ic else np.zeros((0, 9))
        x[self._intr_off: self._ext_off] = intr[:, self._free_intr].reshape(-1)
        ext = np.stack([self.extrinsics_param_[i] for i in self._img_ids]) if self.n_img else np.zeros((0, 6))
        x[self._ext_off: self._ext_off + self.n_img * 3] = ext[:, 0:3].reshape(-1)
        if self._use_disp:
            x[self._disp_off: self._disp_off + 3] = self.disp_param_
        if self._use_tlw:
            x[self._tlw_off: self._tlw_off + 6] = self.tlw_param_
        if self.n_ray:
            x[self.n_cam:] = np.stack([self.rays_param_[t] for t in self._ray_track_ids]).reshape(-1)
        return x

    def _obtain_refined_camera_params(self, cameras: List[Camera]) -> List[List[Ray]]:
        """``PTZRayOptimizer::ObtainRefinedCameraParams``"""
        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue

            ic_id = self.shared_ic_ids_[i]
            intr = self.intrinsics_param_[ic_id]
            ext = self.extrinsics_param_[i]

            param = np.zeros(15, dtype=np.float64)
            param[0] = intr[0]
            param[1] = intr[1] if self.type_ == FactorType.PTZRayFxfyDist else intr[0]
            param[2] = intr[2]
            param[3] = intr[3]
            param[4] = ext[0]
            param[5] = ext[1]
            param[6] = ext[2]
            param[7] = ext[3]
            param[8] = ext[4]
            displacement = (
                self.disp_param_[0] + self.disp_param_[1] * param[0] + self.disp_param_[2] * param[0] * param[0]
            )
            param[9] = ext[5] + displacement
            param[10:15] = intr[4:9]

            cameras[i].from_vector(param)

        # Transform all cameras from local to world: T_i_w = T_i_l * T_l_w
        R_l_w, t_l_w = t_l_w_from_param(self.tlw_param_)

        for i in range(self.num_cams_):
            if not self._is_candidate(i):
                continue
            cameras[i].t = cameras[i].R @ t_l_w + cameras[i].t
            cameras[i].R = cameras[i].R @ R_l_w

        # obtain rays (expressed in world coordinates)
        rays: List[List[Ray]] = [[] for _ in range(self.num_cams_)]
        R_w_l = R_l_w.T
        t_w_l = -R_l_w.T @ t_l_w
        for track_id, track in self.tracks_.items():
            xyz = self.rays_param_[track_id].reshape(3, 1)
            ray_w = (R_w_l @ xyz + t_w_l).reshape(3)
            for image_id, feature_id in track.items():
                uv = self.features_[image_id].keypoints[feature_id]
                rays[image_id].append(Ray(track_id, ray_w.copy(), np.asarray(uv, dtype=np.float64)))

        return rays
