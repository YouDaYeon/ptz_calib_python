#!/usr/bin/env python
"""PLAN.md section 6-3 - evaluation for own data, where no GT camera parameters exist.

The public datasets are scored with FLE/APE/IoU, all of which need GT poses.  Own
data has none, so this reports what *can* be measured without GT poses:

* **cross-frame consistency (px)** - the primary metric available today.  Two
  frames that see the same point must agree about where it is: back-project a
  matched keypoint from frame i to a ray and project it into frame j, then
  compare with the keypoint actually matched there.  Pure rotation means the
  transfer is the infinite homography ``K_j R_j R_i^T K_i^-1``.  This measures
  only the mutual consistency of the estimated cameras, so it is independent of
  3D measurement accuracy - but it cannot detect an error shared by all frames (a global
  focal-length bias, for instance) and it is not an absolute accuracy figure.
* **PTZ readout relative rotation (deg)** - section 6-4 quasi-GT.  Frames sharing
  a tilt differ by a pure pan, so ``angle(R_i R_j^T)`` must equal the pan
  difference.  A best-fit scale is reported alongside the raw error: a constant
  scale points at a readout-vs-focal-length scale mismatch, scattered residuals
  point at a genuine rotation error.  A second, stronger form of the same check
  fits every logged frame at once - ``R = B Rx(st*tilt) Ry(sp*pan) A`` with two
  fixed alignment rotations and one scale per axis - which constrains the pan and
  tilt scales *separately*.  A scale that shows up on both axes cannot be one
  axis' mechanical gear error.
* **per-difficulty aggregation** - section 6-2.  A 5th ``difficulty`` column in the
  PTZ log labels each test frame (e.g. ``boundary``); success rate and consistency
  are then reported per label.
* **registration / relocalization rates**, and the holdout 2D-3D reprojection
  error (``--holdout``): project unused survey points with the saved ``K,R,t``
  and compare to the clicked pixels.  Meaningless on ``--no_georef`` params.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptzcalib import data_io  # noqa: E402
from ptzcalib.quality import (  # noqa: E402
    DEFAULT_DEGRADED_THRESHOLD_M,
    DEFAULT_PASS_THRESHOLD_M,
    CalibrationQuality,
    classify,
)


# --------------------------------------------------------------------------
class Cam:
    __slots__ = ("K", "R", "t", "dist", "res")

    def __init__(
        self, K: np.ndarray, R: np.ndarray, t: np.ndarray, res: Sequence[int], dist: Optional[np.ndarray] = None
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t, dtype=np.float64).reshape(3, 1)
        self.res = res
        self.dist = (
            np.zeros((5, 1), dtype=np.float64)
            if dist is None
            else np.asarray(dist, dtype=np.float64).reshape(-1)[:5].reshape(5, 1)
        )


def _camera_dist(value: dict) -> np.ndarray:
    raw = np.asarray(value.get("dist") or [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1)
    if raw.size < 5:
        raw = np.pad(raw, (0, 5 - raw.size))
    return raw[:5].reshape(5, 1)


def load_cameras(path: str) -> Dict[str, Cam]:
    with open(path, "r", encoding="utf-8") as fin:
        j = json.load(fin)
    out: Dict[str, Cam] = {}
    for name, v in j["cameras"].items():
        out[name] = Cam(
            np.asarray(v["K"], dtype=np.float64).reshape(3, 3),
            np.asarray(v["R"], dtype=np.float64).reshape(3, 3),
            np.asarray(v["t"], dtype=np.float64).reshape(3, 1),
            v["res"],
            _camera_dist(v),
        )
    return out


def load_features(feature_dir: str, fname: str) -> np.ndarray:
    kpts, _desc = data_io.read_colmap_features(os.path.join(feature_dir, fname + ".txt"))
    return kpts


def load_blocks(matches_path: str) -> List[Tuple[str, str, np.ndarray]]:
    pairs, names = data_io.read_colmap_matches(matches_path)
    return [(names[i][0], names[i][1], pairs[i]) for i in range(len(pairs))]


def stem(fname: str) -> str:
    return os.path.splitext(fname)[0]


# --------------------------------------------------------------------------
def transfer_errors(
    kpts_i: np.ndarray, kpts_j: np.ndarray, matches: np.ndarray, cam_i: Cam, cam_j: Cam
) -> np.ndarray:
    """Symmetric transfer error in px through the infinite homography.

    Distortion is deliberately ignored: ``PTZRayFactor`` does not apply it
    either, so this measures the same model the optimiser actually fitted.
    """
    if matches.shape[0] == 0:
        return np.zeros((0,))

    H_ij = cam_j.K @ cam_j.R @ cam_i.R.T @ np.linalg.inv(cam_i.K)
    H_ji = np.linalg.inv(H_ij)

    pi = kpts_i[matches[:, 0]]
    pj = kpts_j[matches[:, 1]]

    def warp(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1) @ H.T
        w = h[:, 2:3]
        w[np.abs(w) < 1e-12] = 1e-12
        return h[:, :2] / w

    e_fwd = np.linalg.norm(warp(H_ij, pi) - pj, axis=1)
    e_bwd = np.linalg.norm(warp(H_ji, pj) - pi, axis=1)
    return 0.5 * (e_fwd + e_bwd)


def summarise(values: np.ndarray, label: str) -> dict:
    if values.size == 0:
        return {"label": label, "count": 0}
    return {
        "label": label,
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "rmse": float(np.sqrt((values**2).mean())),
    }


_T_EPS = 1e-6


def _click_pixels(marker: dict, res: Sequence[int]) -> np.ndarray:
    """Clicked pixels: ``pix_raw`` if present, else denormalised ``pix`` (W,H)."""
    raw = marker.get("pix_raw") or []
    if len(raw):
        return np.asarray(raw, dtype=np.float64).reshape(-1, 2)
    pix = np.asarray(marker.get("pix") or [], dtype=np.float64).reshape(-1, 2)
    if pix.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    width, height = int(res[0]), int(res[1])
    uv = np.empty_like(pix)
    uv[:, 0] = (width * pix[:, 0]).astype(np.float32)
    uv[:, 1] = (height * pix[:, 1]).astype(np.float32)
    return uv.astype(np.float64)


def _click_ray_on_plane(cam: Cam, uv: np.ndarray, z_plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project pixels to rays and intersect the horizontal planes ``Z = z_plane``.

    2D→3D is otherwise underdetermined; survey points lie on the floor so this is
    the inverse of ``projectPoints`` onto that plane.  Returns ``(xyz, valid)``.
    """
    n = uv.shape[0]
    dist = cam.dist.reshape(-1)
    # normalised undistorted coords: ray in camera frame is (x, y, 1)
    undist = cv2.undistortPoints(uv.reshape(-1, 1, 2), cam.K, dist).reshape(-1, 2)
    ray_cam = np.concatenate([undist, np.ones((n, 1))], axis=1)
    ray_w = (cam.R.T @ ray_cam.T).T
    C = (-cam.R.T @ cam.t).ravel()
    dz = ray_w[:, 2]
    valid = np.abs(dz) > 1e-12
    lam = np.zeros(n, dtype=np.float64)
    lam[valid] = (z_plane[valid] - C[2]) / dz[valid]
    xyz = C[None, :] + lam[:, None] * ray_w
    x_cam = (cam.R @ xyz.T + cam.t).T
    valid = valid & (lam > 0) & (x_cam[:, 2] > 0.0)
    return xyz, valid


def holdout_2d3d(cams: Dict[str, Cam], holdout_path: str) -> dict:
    """Section 6-3: holdout survey vs saved ``K,R,t``.

    Pixel error: project ``marker.pos`` and compare to the click.
    Metre error: shoot a ray from the click, intersect the point's horizontal
    plane (``Z = pos_z``, usually 0), and compare to the survey 3D point.
    ``--no_georef`` cameras (``t = 0``) are skipped.
    """
    if not os.path.isfile(holdout_path):
        return {"status": "error", "reason": f"holdout file not found: {holdout_path}"}
    try:
        with open(holdout_path, "r", encoding="utf-8") as fin:
            spec_cams = json.load(fin)["cameras"]
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {"status": "error", "reason": f"failed to read holdout JSON: {exc}"}

    skipped_frames = sorted(n for n in spec_cams if n not in cams)
    matched = sorted(n for n in spec_cams if n in cams)
    if not matched:
        return {
            "status": "error",
            "reason": "no holdout frames overlap registered cameras",
            "skipped_frames": skipped_frames,
        }
    if all(float(np.linalg.norm(cams[n].t)) < _T_EPS for n in matched):
        return {
            "status": "skipped",
            "reason": "cameras have no world translation "
            "(run georeferencing first; --no_georef params cannot be scored)",
            "skipped_frames": skipped_frames,
        }

    all_err: List[np.ndarray] = []
    all_dx: List[np.ndarray] = []
    all_dy: List[np.ndarray] = []
    all_err_m: List[np.ndarray] = []
    all_dx_m: List[np.ndarray] = []
    all_dy_m: List[np.ndarray] = []
    per_frame: List[dict] = []
    n_behind = 0
    n_no_plane = 0

    for name in matched:
        spec = spec_cams[name]
        marker = spec.get("marker") or {}
        pts3d = np.asarray(marker.get("pos") or [], dtype=np.float64).reshape(-1, 3)
        uv = _click_pixels(marker, spec.get("res") or cams[name].res)
        n = min(pts3d.shape[0], uv.shape[0])
        if n == 0:
            per_frame.append(
                {
                    "frame": name,
                    "n": 0,
                    "n_behind_camera": 0,
                    "n_no_plane_intersect": 0,
                    "errors_px": summarise(np.zeros((0,)), name),
                    "errors_m": summarise(np.zeros((0,)), name),
                }
            )
            continue

        pts3d, uv = pts3d[:n], uv[:n]
        cam = cams[name]
        x_cam = (cam.R @ pts3d.T + cam.t).T
        front = x_cam[:, 2] > 0.0
        n_behind += int(np.count_nonzero(~front))
        pts3d, uv = pts3d[front], uv[front]
        if pts3d.shape[0] == 0:
            per_frame.append(
                {
                    "frame": name,
                    "n": 0,
                    "n_behind_camera": int(np.count_nonzero(~front)),
                    "n_no_plane_intersect": 0,
                    "errors_px": summarise(np.zeros((0,)), name),
                    "errors_m": summarise(np.zeros((0,)), name),
                }
            )
            continue

        rvec, _ = cv2.Rodrigues(cam.R)
        pred, _ = cv2.projectPoints(pts3d, rvec, cam.t, cam.K, cam.dist.reshape(-1))
        pred = pred.reshape(-1, 2)
        delta = pred - uv
        err = np.linalg.norm(delta, axis=1)
        dx, dy = delta[:, 0], delta[:, 1]
        all_err.append(err)
        all_dx.append(dx)
        all_dy.append(dy)

        xyz_hat, on_plane = _click_ray_on_plane(cam, uv, pts3d[:, 2])
        n_miss = int(np.count_nonzero(~on_plane))
        n_no_plane += n_miss
        if np.any(on_plane):
            d3 = xyz_hat[on_plane] - pts3d[on_plane]
            err_m = np.linalg.norm(d3, axis=1)
            dx_m, dy_m = d3[:, 0], d3[:, 1]
            all_err_m.append(err_m)
            all_dx_m.append(dx_m)
            all_dy_m.append(dy_m)
            errors_m = summarise(err_m, name)
            mean_abs_dx_m = float(np.abs(dx_m).mean())
            mean_abs_dy_m = float(np.abs(dy_m).mean())
        else:
            errors_m = summarise(np.zeros((0,)), name)
            mean_abs_dx_m = mean_abs_dy_m = float("nan")

        per_frame.append(
            {
                "frame": name,
                "n": int(err.size),
                "n_behind_camera": int(np.count_nonzero(~front)),
                "n_no_plane_intersect": n_miss,
                "errors_px": summarise(err, name),
                "errors_m": errors_m,
                "mean_abs_dx": float(np.abs(dx).mean()),
                "mean_abs_dy": float(np.abs(dy).mean()),
                "mean_abs_dx_m": mean_abs_dx_m,
                "mean_abs_dy_m": mean_abs_dy_m,
            }
        )

    flat = np.concatenate(all_err) if all_err else np.zeros((0,))
    dx_all = np.concatenate(all_dx) if all_dx else np.zeros((0,))
    dy_all = np.concatenate(all_dy) if all_dy else np.zeros((0,))
    flat_m = np.concatenate(all_err_m) if all_err_m else np.zeros((0,))
    dx_m_all = np.concatenate(all_dx_m) if all_dx_m else np.zeros((0,))
    dy_m_all = np.concatenate(all_dy_m) if all_dy_m else np.zeros((0,))
    out: dict = {
        "status": "ok",
        "n_frames": sum(1 for p in per_frame if p["n"] > 0),
        "n_obs": int(flat.size),
        "n_behind_camera": n_behind,
        "n_no_plane_intersect": n_no_plane,
        "skipped_frames": skipped_frames,
        "errors_px": summarise(flat, "holdout 2d-3d px"),
        "errors_m": summarise(flat_m, "holdout 2d-3d m"),
        "per_frame": per_frame,
    }
    if flat.size:
        out["mean_abs_dx"] = float(np.abs(dx_all).mean())
        out["mean_abs_dy"] = float(np.abs(dy_all).mean())
    if flat_m.size:
        out["mean_abs_dx_m"] = float(np.abs(dx_m_all).mean())
        out["mean_abs_dy_m"] = float(np.abs(dy_m_all).mean())
    return out


# --------------------------------------------------------------------------
class PtzEntry:
    """One row of the PTZ log: angles plus an optional difficulty label."""

    __slots__ = ("pan", "tilt", "zoom", "difficulty")

    def __init__(self, pan: float, tilt: float, zoom: float, difficulty: str) -> None:
        self.pan, self.tilt, self.zoom, self.difficulty = pan, tilt, zoom, difficulty

    def __getitem__(self, i: int) -> float:  # keeps the original tuple-style access working
        return (self.pan, self.tilt, self.zoom)[i]


def load_ptz_log(ptz_log_path: str) -> Dict[str, PtzEntry]:
    """Parse ``fname,pan,tilt,zoom[,difficulty]``.  ``#`` lines are comments.

    The difficulty label is optional and must not contain whitespace or commas.
    """
    log: Dict[str, PtzEntry] = {}
    with open(ptz_log_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(",", " ").split()]
            if len(parts) < 4 or parts[0].lower() == "fname":
                continue
            log[stem(parts[0])] = PtzEntry(
                float(parts[1]), float(parts[2]), float(parts[3]), parts[4] if len(parts) > 4 else ""
            )
    return log


def _rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _geodesic_deg(A: np.ndarray, B: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(A @ B.T)) - 1.0) / 2.0))))


def ptz_pose_fit(cams: Dict[str, Cam], log: Dict[str, PtzEntry]) -> Optional[dict]:
    """Section 6-4, absolute form: fit every logged pose at once.

    Model ``R_i = B @ Rx(st * tilt_i) @ Ry(sp * pan_i) @ A``.  ``A`` (world -> pan
    frame) and ``B`` (tilt frame -> camera) are fixed unknown rotations, so the two
    scales ``sp`` / ``st`` are the only things that can absorb a readout-vs-estimate
    angular mismatch - and they are estimated per axis.  Reported alongside is the
    residual with both scales pinned to 1, which is what tells you whether the
    scale factor is real or just soaking up noise.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415 - optional heavy import
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    names = [n for n in sorted(log) if n in cams]
    if len(names) < 4:
        return None

    pan = np.asarray([log[n].pan for n in names])
    tilt = np.asarray([log[n].tilt for n in names])
    R_est = np.stack([cams[n].R for n in names])
    pan_c, tilt_c = pan - pan.mean(), tilt - tilt.mean()

    def model(par: np.ndarray, fix_scale: bool) -> np.ndarray:
        B = Rotation.from_rotvec(par[0:3]).as_matrix()
        A = Rotation.from_rotvec(par[3:6]).as_matrix()
        sp, st = (1.0, 1.0) if fix_scale else (par[6], par[7])
        return np.stack([B @ _rot_x(st * t) @ _rot_y(sp * p) @ A for p, t in zip(pan_c, tilt_c)])

    def solve(fix_scale: bool) -> Tuple[np.ndarray, np.ndarray]:
        def resid(par: np.ndarray) -> np.ndarray:
            M = model(par, fix_scale)
            return Rotation.from_matrix(np.einsum("nij,nkj->nik", R_est, M)).as_rotvec(degrees=True).ravel()

        rng = np.random.default_rng(0)  # fixed seed: the multi-start must be reproducible
        best = None
        n_par = 6 if fix_scale else 8
        for _ in range(14):
            x0 = np.zeros(n_par)
            x0[:6] = rng.normal(0.0, 1.6, 6)
            if not fix_scale:
                x0[6] = float(rng.choice([1.0, -1.0]))
                x0[7] = float(rng.choice([1.0, -1.0]))
            sol = least_squares(resid, x0, xtol=1e-13, ftol=1e-13, max_nfev=2000)
            if best is None or sol.cost < best.cost:
                best = sol
        assert best is not None
        ang = np.asarray([_geodesic_deg(Re, Rm) for Re, Rm in zip(R_est, model(best.x, fix_scale))])
        return best.x, ang

    par, ang = solve(False)
    _, ang_unit = solve(True)

    return {
        "frames": len(names),
        # sign only reflects the axis convention of the readout; the magnitude is the finding
        "pan_scale": abs(float(par[6])),
        "tilt_scale": abs(float(par[7])),
        "pan_span_deg": float(pan.max() - pan.min()),
        "tilt_span_deg": float(tilt.max() - tilt.min()),
        "residual_deg": {
            "mean": float(ang.mean()),
            "median": float(np.median(ang)),
            "max": float(ang.max()),
        },
        "residual_unit_scale_deg": {
            "mean": float(ang_unit.mean()),
            "median": float(np.median(ang_unit)),
            "max": float(ang_unit.max()),
        },
        "worst_frames": [
            {"frame": names[i], "residual_deg": float(ang[i]), "pan": pan[i], "tilt": tilt[i]}
            for i in np.argsort(-ang)[:5]
        ],
    }


def ptz_rotation_check(cams: Dict[str, Cam], ptz_log_path: str) -> Optional[dict]:
    """Section 6-4: angle(R_i R_j^T) vs the pan readout, for frames sharing a tilt."""
    log = load_ptz_log(ptz_log_path)

    usable = [n for n in log if n in cams]
    if len(usable) < 2:
        return {"usable_frames": len(usable), "note": "not enough frames with both a readout and a pose"}

    rows = []
    ref_ang, est_ang = [], []
    for a, b in itertools.combinations(sorted(usable), 2):
        if abs(log[a][1] - log[b][1]) > 1e-6:
            continue  # different tilt: the pure-pan identity no longer holds
        dR = cams[a].R @ cams[b].R.T
        est = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0))))
        d = abs(log[a][0] - log[b][0]) % 360.0
        ref = min(d, 360.0 - d)
        rows.append({"pair": [a, b], "readout_deg": ref, "estimated_deg": est, "diff_deg": est - ref})
        ref_ang.append(ref)
        est_ang.append(est)

    if not rows:
        return {"usable_frames": len(usable), "note": "no same-tilt pairs"}

    x = np.asarray(ref_ang)
    y = np.asarray(est_ang)
    diff = np.abs(y - x)
    scale = float((x @ y) / (x @ x)) if float(x @ x) > 0 else float("nan")
    resid = np.abs(y - scale * x)

    out = {
        "usable_frames": len(usable),
        "pairs": len(rows),
        "abs_diff_deg": {"mean": float(diff.mean()), "median": float(np.median(diff)), "max": float(diff.max())},
        "best_fit_scale": scale,
        "residual_after_scale_deg": {"mean": float(resid.mean()), "max": float(resid.max())},
        "detail": rows,
    }
    fit = ptz_pose_fit(cams, log)
    if fit is not None:
        out["pose_fit"] = fit
    return out


def focal_vs_zoom(cams: Dict[str, Cam], ptz_log_path: str) -> Optional[dict]:
    """Section 5: estimated ``f`` must rise monotonically with the zoom readout."""
    zoom = {n: e.zoom for n, e in load_ptz_log(ptz_log_path).items() if n in cams}
    if len(zoom) < 2:
        return None

    groups: Dict[float, List[float]] = {}
    for name, z in zoom.items():
        groups.setdefault(z, []).append(float(cams[name].K[0, 0]))

    levels = sorted(groups)
    means = [float(np.mean(groups[z])) for z in levels]
    monotonic = all(means[i] < means[i + 1] for i in range(len(means) - 1))

    out = {
        "levels": [
            {
                "zoom": z,
                "frames": len(groups[z]),
                "f_mean": m,
                "f_min": min(groups[z]),
                "f_max": max(groups[z]),
                # constant across levels iff the zoom readout is proportional to f
                "f_over_zoom": m / z,
            }
            for z, m in zip(levels, means)
        ],
        "monotonic_increasing": monotonic,
    }
    ratios = [m / z for z, m in zip(levels, means)]
    out["f_over_zoom_spread_pct"] = 100.0 * (max(ratios) / min(ratios) - 1.0)
    if len(levels) == 2:
        out["f_ratio"] = means[1] / means[0]
        out["zoom_ratio"] = levels[1] / levels[0]
        out["ratio_discrepancy_pct"] = 100.0 * abs(out["f_ratio"] / out["zoom_ratio"] - 1.0)
    return out


# --------------------------------------------------------------------------
def eval_offline(args: argparse.Namespace) -> dict:
    cams = load_cameras(args.params)
    fnames = [f for f in sorted(os.listdir(args.images)) if f != "mask.png" and stem(f) not in ("",)]
    fnames = [f for f in fnames if os.path.splitext(f)[1].lower() in data_io.VALID_IMG_EXTS]

    feats = {f: load_features(args.features, f) for f in fnames if stem(f) in cams}
    blocks = load_blocks(os.path.join(args.features, "pairs_matches.txt"))

    per_pair = []
    all_err: List[np.ndarray] = []
    seen = set()
    for na, nb, matches in blocks:
        sa, sb = stem(na), stem(nb)
        if sa not in cams or sb not in cams:
            continue
        key = tuple(sorted((sa, sb)))
        if key in seen:  # offline blocks are bidirectional
            continue
        seen.add(key)
        err = transfer_errors(feats[na], feats[nb], matches, cams[sa], cams[sb])
        if err.size:
            all_err.append(err)
            per_pair.append({"pair": [sa, sb], "n": int(err.size), "mean_px": float(err.mean())})

    flat = np.concatenate(all_err) if all_err else np.zeros((0,))
    pair_means = np.asarray([p["mean_px"] for p in per_pair])

    out = {
        "mode": "offline",
        "registered": len(cams),
        "total_images": len(fnames),
        "registration_rate": len(cams) / max(len(fnames), 1),
        "unregistered": sorted(stem(f) for f in fnames if stem(f) not in cams),
        "pairs_evaluated": len(per_pair),
        "cross_frame_consistency_px": summarise(flat, "all matched observations"),
        "per_pair_mean_px": summarise(pair_means, "per-pair mean"),
        "worst_pairs": sorted(per_pair, key=lambda p: -p["mean_px"])[:5],
    }
    if args.ptz_log:
        out["ptz_rotation_check"] = ptz_rotation_check(cams, args.ptz_log)
        out["focal_vs_zoom"] = focal_vs_zoom(cams, args.ptz_log)
    return out


def eval_online(args: argparse.Namespace) -> dict:
    ref_cams = load_cameras(args.ref_params)
    test_cams = load_cameras(args.test_params)

    test_fnames = [
        f
        for f in sorted(os.listdir(args.test_images))
        if f != "mask.png" and os.path.splitext(f)[1].lower() in data_io.VALID_IMG_EXTS
    ]
    blocks = load_blocks(os.path.join(args.test_features, "pairs_matches.txt"))

    # mirror run_ptz_reloc.find_best_match: the reference with the most matches wins
    best: Dict[str, Tuple[str, np.ndarray]] = {}
    for nref, ntest, matches in blocks:
        cur = best.get(ntest)
        if cur is None or matches.shape[0] > cur[1].shape[0]:
            best[ntest] = (nref, matches)

    per_frame = []
    all_err: List[np.ndarray] = []
    for ntest in test_fnames:
        if stem(ntest) not in test_cams or ntest not in best:
            continue
        nref, matches = best[ntest]
        if stem(nref) not in ref_cams:
            continue
        kt = load_features(args.test_features, ntest)
        kr = load_features(args.ref_features, nref)
        err = transfer_errors(kr, kt, matches, ref_cams[stem(nref)], test_cams[stem(ntest)])
        if err.size:
            all_err.append(err)
            per_frame.append(
                {
                    "test": stem(ntest),
                    "ref": stem(nref),
                    "n": int(err.size),
                    "mean_px": float(err.mean()),
                    "median_px": float(np.median(err)),
                }
            )

    flat = np.concatenate(all_err) if all_err else np.zeros((0,))
    frame_means = np.asarray([p["mean_px"] for p in per_frame])
    failed = sorted(stem(f) for f in test_fnames if stem(f) not in test_cams)

    out = {
        "mode": "online",
        "total_test_frames": len(test_fnames),
        "relocalized": len(test_cams),
        "success_rate": len(test_cams) / max(len(test_fnames), 1),
        "failed_frames": failed,
        "frames_evaluated": len(per_frame),
        "cross_frame_consistency_px": summarise(flat, "all matched observations"),
        "per_frame_mean_px": summarise(frame_means, "per-frame mean"),
        "worst_frames": sorted(per_frame, key=lambda p: -p["mean_px"])[:5],
        "per_frame": per_frame,
    }
    if args.ptz_log:
        by_diff = difficulty_breakdown(load_ptz_log(args.ptz_log), test_fnames, test_cams, per_frame)
        if by_diff is not None:
            out["per_difficulty"] = by_diff
        # the same section 5 / 6-4 checks as offline, on the relocalized poses
        out["ptz_rotation_check"] = ptz_rotation_check(test_cams, args.ptz_log)
        out["focal_vs_zoom"] = focal_vs_zoom(test_cams, args.ptz_log)
    return out


def difficulty_breakdown(
    log: Dict[str, PtzEntry],
    test_fnames: Sequence[str],
    test_cams: Dict[str, Cam],
    per_frame: Sequence[dict],
) -> Optional[List[dict]]:
    """Section 6-2: success rate and consistency split by the log's difficulty label."""
    labels = {n: e.difficulty for n, e in log.items() if e.difficulty}
    if not labels:
        return None

    mean_by_frame = {p["test"]: p["mean_px"] for p in per_frame}
    groups: Dict[str, List[str]] = {}
    for f in test_fnames:
        lab = labels.get(stem(f))
        if lab:
            groups.setdefault(lab, []).append(stem(f))

    rows = []
    for lab in sorted(groups):
        frames = groups[lab]
        ok = [n for n in frames if n in test_cams]
        means = np.asarray([mean_by_frame[n] for n in ok if n in mean_by_frame])
        rows.append(
            {
                "difficulty": lab,
                "frames": len(frames),
                "relocalized": len(ok),
                "success_rate": len(ok) / len(frames),
                "failed": sorted(n for n in frames if n not in test_cams),
                "consistency_px": summarise(means, f"{lab}: per-frame mean"),
            }
        )
    unlabelled = [stem(f) for f in test_fnames if stem(f) not in labels]
    if unlabelled:
        rows.append({"difficulty": "(unlabelled)", "frames": len(unlabelled), "frame_names": unlabelled})
    return rows


# --------------------------------------------------------------------------
def _verdict_reason(
    grade: CalibrationQuality,
    rmse_m: Optional[float],
    holdout: dict,
    pass_m: float,
    degraded_m: float,
) -> str:
    if grade is CalibrationQuality.UNKNOWN:
        detail = holdout.get("reason") or ""
        if not detail:
            status = holdout.get("status", "missing")
            detail = (
                "no holdout observation survived the validity checks"
                if status == "ok"
                else f"holdout status: {status}"
            )
        return f"no metre RMSE available - {detail}"
    if grade is CalibrationQuality.VALID:
        return f"rmse {rmse_m:.4f} m <= pass threshold {pass_m} m"
    if grade is CalibrationQuality.DEGRADED:
        return f"pass threshold {pass_m} m < rmse {rmse_m:.4f} m <= degraded limit {degraded_m} m"
    return f"rmse {rmse_m:.4f} m > degraded limit {degraded_m} m"


def build_verdict(report: dict, pass_m: float, degraded_m: float) -> dict:
    """Grade the run against the project KPI and tag every holdout frame.

    Only ``holdout_2d3d.errors_m.rmse`` decides the grade - that metre RMSE *is*
    the KPI.  Observation count, registration rate and the exclusion counters are
    recorded next to it but never move the grade, so ``calibration_quality``
    always means one thing and one thing only.  Read them together: a ``valid``
    grade backed by nine observations out of three frames says less than the same
    grade backed by thirty.

    ``FALLBACK``/``MOVING`` cannot arise here; they are runtime states that only
    ``live_web`` can observe (see ``ptzcalib/quality.py``).
    """
    holdout = report.get("holdout_2d3d") or {}
    errors_m = holdout.get("errors_m") or {}
    rmse_m = errors_m.get("rmse") if errors_m.get("count") else None

    # per-frame grade: each holdout frame is one camera of the multi-camera set
    for frame in holdout.get("per_frame") or []:
        frame_m = frame.get("errors_m") or {}
        frame_rmse = frame_m.get("rmse") if frame_m.get("count") else None
        frame["quality"] = classify(frame_rmse, pass_m, degraded_m).value

    if report["mode"] == "offline":
        rate, ok, total = (
            report.get("registration_rate"),
            report.get("registered"),
            report.get("total_images"),
        )
    else:
        rate, ok, total = (
            report.get("success_rate"),
            report.get("relocalized"),
            report.get("total_test_frames"),
        )

    grade = classify(rmse_m, pass_m, degraded_m)
    return {
        "metric": "holdout_2d3d.errors_m.rmse",
        "pass_threshold_m": pass_m,
        "degraded_threshold_m": degraded_m,
        "rmse_m": rmse_m,
        "quality": grade.value,
        "reason": _verdict_reason(grade, rmse_m, holdout, pass_m, degraded_m),
        # --- reported alongside the grade, deliberately not part of it -------
        "n_obs": int(errors_m.get("count") or 0),
        "n_frames_measured": int(holdout.get("n_frames") or 0),
        "registered": ok,
        "total": total,
        "rate": rate,
        "n_behind_camera": holdout.get("n_behind_camera"),
        "n_no_plane_intersect": holdout.get("n_no_plane_intersect"),
        "skipped_frames": holdout.get("skipped_frames") or [],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTZ-Calib evaluation for own data (no GT poses)")
    p.add_argument("--mode", choices=["offline", "online"], required=True)
    p.add_argument("-i", "--images", help="offline: reference images directory")
    p.add_argument("-f", "--features", help="offline: features/matches directory")
    p.add_argument("--params", help="offline: run_ptz_ba.py output JSON")
    p.add_argument("--ref_images", help="online: reference images directory")
    p.add_argument("--ref_features", help="online: reference features directory")
    p.add_argument("--ref_params", help="online: reference camera JSON")
    p.add_argument("--test_images", help="online: test images directory")
    p.add_argument("--test_features", help="online: test features/matches directory")
    p.add_argument("--test_params", help="online: run_ptz_reloc.py output JSON")
    p.add_argument("--ptz_log", default="", help="CSV fname,pan,tilt,zoom (section 6-4 quasi-GT)")
    p.add_argument(
        "--holdout",
        default="",
        help="annotation JSON with holdout 2D-3D points, e.g. data/infiniq/gt/offline.json "
             "- holdout 2D-3D error (section 6-3 primary metric)",
    )
    p.add_argument(
        "--pass_threshold_m",
        type=float,
        default=DEFAULT_PASS_THRESHOLD_M,
        help="valid/degraded boundary for the holdout 2D-3D metre RMSE, i.e. the "
             f"project KPI (default: {DEFAULT_PASS_THRESHOLD_M})",
    )
    p.add_argument(
        "--degraded_threshold_m",
        type=float,
        default=DEFAULT_DEGRADED_THRESHOLD_M,
        help="degraded/invalid boundary - the upper limit of the degraded band "
             f"(default: {DEFAULT_DEGRADED_THRESHOLD_M})",
    )
    p.add_argument("-o", "--output", default="", help="write the report as JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.degraded_threshold_m < args.pass_threshold_m:
        raise SystemExit(
            f"--degraded_threshold_m ({args.degraded_threshold_m}) is below "
            f"--pass_threshold_m ({args.pass_threshold_m}): the degraded band would be empty"
        )

    if args.mode == "offline":
        missing = [k for k in ("images", "features", "params") if not getattr(args, k)]
        if missing:
            raise SystemExit(f"--mode offline requires: {', '.join('--' + m for m in missing)}")
        report = eval_offline(args)
    else:
        need = ("ref_images", "ref_features", "ref_params", "test_images", "test_features", "test_params")
        missing = [k for k in need if not getattr(args, k)]
        if missing:
            raise SystemExit(f"--mode online requires: {', '.join('--' + m for m in missing)}")
        report = eval_online(args)

    if args.holdout:
        holdout_cams = load_cameras(args.params if args.mode == "offline" else args.test_params)
        report["holdout_2d3d"] = holdout_2d3d(holdout_cams, args.holdout)
    else:
        report["holdout_2d3d"] = {
            "status": "skipped",
            "reason": "no --holdout annotation JSON given; this is the primary metric and stays unavailable",
        }

    report["verdict"] = build_verdict(report, args.pass_threshold_m, args.degraded_threshold_m)
    report["calibration_quality"] = report["verdict"]["quality"]

    print("=" * 66)
    print(f"eval_own - {report['mode']}")
    print("=" * 66)
    if report["mode"] == "offline":
        print(f"registered        : {report['registered']}/{report['total_images']} "
              f"({100 * report['registration_rate']:.1f}%)")
        if report["unregistered"]:
            print(f"unregistered      : {report['unregistered']}")
        print(f"pairs evaluated   : {report['pairs_evaluated']}")
    else:
        print(f"relocalized       : {report['relocalized']}/{report['total_test_frames']} "
              f"({100 * report['success_rate']:.1f}%)")
        print(f"frames evaluated  : {report['frames_evaluated']}")
        if report["failed_frames"]:
            print(f"failed            : {len(report['failed_frames'])} frames")

    c = report["cross_frame_consistency_px"]
    if c["count"]:
        print(
            f"cross-frame (px)  : mean {c['mean']:.2f}  median {c['median']:.2f}  "
            f"p90 {c['p90']:.2f}  rmse {c['rmse']:.2f}   [n={c['count']}]"
        )

    if report.get("focal_vs_zoom"):
        fz = report["focal_vs_zoom"]
        print(f"f vs zoom         : monotonic={fz['monotonic_increasing']}")
        for lv in fz["levels"]:
            print(
                f"    zoom X{lv['zoom']:.1f}  f_mean {lv['f_mean']:8.1f}  ({lv['frames']} frames)"
                f"  f/zoom {lv['f_over_zoom']:7.1f}"
            )
        if "f_over_zoom_spread_pct" in fz:
            print(f"    f/zoom spread across levels: {fz['f_over_zoom_spread_pct']:.2f}%")
        if "ratio_discrepancy_pct" in fz:
            print(
                f"    f ratio {fz['f_ratio']:.4f} vs zoom ratio {fz['zoom_ratio']:.4f} "
                f"-> {fz['ratio_discrepancy_pct']:.2f}% discrepancy"
            )

    if report.get("ptz_rotation_check") and "abs_diff_deg" in report["ptz_rotation_check"]:
        pr = report["ptz_rotation_check"]
        print(
            f"PTZ rotation      : {pr['pairs']} pairs, abs diff mean {pr['abs_diff_deg']['mean']:.2f} deg "
            f"(max {pr['abs_diff_deg']['max']:.2f})"
        )
        print(
            f"    best-fit scale {pr['best_fit_scale']:.4f} -> residual mean "
            f"{pr['residual_after_scale_deg']['mean']:.3f} deg (max {pr['residual_after_scale_deg']['max']:.3f})"
        )
        if pr.get("pose_fit"):
            pf = pr["pose_fit"]
            print(
                f"    full pose fit ({pf['frames']} frames): pan scale {pf['pan_scale']:.4f} "
                f"(span {pf['pan_span_deg']:.1f} deg), tilt scale {pf['tilt_scale']:.4f} "
                f"(span {pf['tilt_span_deg']:.1f} deg)"
            )
            print(
                f"      residual mean {pf['residual_deg']['mean']:.3f} / max {pf['residual_deg']['max']:.3f} deg"
                f"   | scales pinned to 1: mean {pf['residual_unit_scale_deg']['mean']:.3f} / "
                f"max {pf['residual_unit_scale_deg']['max']:.3f} deg"
            )

    if report.get("per_difficulty"):
        print("per difficulty    :")
        for row in report["per_difficulty"]:
            if "consistency_px" not in row:
                print(f"    {row['difficulty']:<14} {row['frames']} frames")
                continue
            cc = row["consistency_px"]
            extra = (
                f"consistency mean {cc['mean']:.2f} median {cc['median']:.2f} max {cc['max']:.2f} px"
                if cc["count"]
                else "no consistency samples"
            )
            print(
                f"    {row['difficulty']:<14} {row['relocalized']:>2}/{row['frames']:<2} "
                f"({100 * row['success_rate']:5.1f}%)  {extra}"
            )

    h = report["holdout_2d3d"]
    if h["status"] == "ok" and h["errors_px"].get("count"):
        e = h["errors_px"]
        print(
            f"holdout 2d-3d     : rmse {e['rmse']:.2f}  mean {e['mean']:.2f}  "
            f"median {e['median']:.2f}  p90 {e['p90']:.2f} px  [n={e['count']}]"
        )
        if "mean_abs_dx" in h:
            print(f"    |dx| {h['mean_abs_dx']:.2f}  |dy| {h['mean_abs_dy']:.2f} px")
        em = h.get("errors_m") or {}
        if em.get("count"):
            print(
                f"    metres        : rmse {em['rmse']:.4f}  mean {em['mean']:.4f}  "
                f"median {em['median']:.4f}  p90 {em['p90']:.4f} m"
                f"  ({100 * em['rmse']:.1f} / {100 * em['mean']:.1f} cm rmse/mean)"
            )
            if "mean_abs_dx_m" in h:
                print(f"    |dX| {h['mean_abs_dx_m']:.4f}  |dY| {h['mean_abs_dy_m']:.4f} m")
        for fr in h["per_frame"]:
            pe = fr["errors_px"]
            if not pe.get("count"):
                continue
            line = (
                f"    {fr['frame']}: n={pe['count']}  rms {pe['rmse']:.2f} px  "
                f"mean {pe['mean']:.2f}  |dx| {fr['mean_abs_dx']:.2f}  |dy| {fr['mean_abs_dy']:.2f}"
            )
            pm = fr.get("errors_m") or {}
            if pm.get("count"):
                line += (
                    f"  |  rms {pm['rmse']:.4f} m ({100 * pm['rmse']:.1f} cm)"
                    f"  |dX| {fr['mean_abs_dx_m']:.4f}  |dY| {fr['mean_abs_dy_m']:.4f}"
                    f"  [{fr.get('quality', '-')}]"
                )
            print(line)
    else:
        extra = f" ({h['reason']})" if h.get("reason") else ""
        print(f"holdout 2d-3d     : {h['status']}{extra}")

    v = report["verdict"]
    if v["rmse_m"] is not None:
        print(
            f"calib quality     : {v['quality'].upper()}  rmse {v['rmse_m']:.4f} m  "
            f"(valid <= {v['pass_threshold_m']}, degraded <= {v['degraded_threshold_m']} m)"
        )
    else:
        print(f"calib quality     : {v['quality'].upper()}  ({v['reason']})")
    parts = [f"n_obs {v['n_obs']} over {v['n_frames_measured']} frames"]
    if v["rate"] is not None:
        parts.append(f"registered {v['registered']}/{v['total']} ({100 * v['rate']:.1f}%)")
    if v["n_behind_camera"] is not None:
        excluded = f"excluded {v['n_behind_camera']} behind / {v['n_no_plane_intersect']} off-plane"
        if v["skipped_frames"]:
            excluded += f" / {len(v['skipped_frames'])} unregistered frames"
        parts.append(excluded)
    print("    not graded    : " + "  |  ".join(parts))
    print("=" * 66)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fout:
            json.dump(report, fout, indent=2)
            fout.write("\n")
        print(f"report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
