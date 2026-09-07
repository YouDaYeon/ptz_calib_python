"""Online stage (PTZ-Reloc, ``--dist``) applied to a single live frame.

``scripts/run_ptz_reloc.py`` reads a directory of test images plus a
pre-computed ``pairs_matches.txt``.  Live frames have neither, so this module
does the two halves that the offline tooling splits between
``scripts/extract_match.py --mode online`` and ``run_ptz_reloc.py``:

1. SuperPoint on the live frame (masked exactly like the reference set),
   SuperGlue against every registered reference, confidence gate + RANSAC
   homography - i.e. ``extract_match.filter_pair``;
2. the reference with the most inliers wins (``run_ptz_reloc.FindBestMatch``)
   and drives ``KRTOptimizer`` with ``FactorType.FDist``.

The matcher classes are imported from ``scripts/extract_match.py`` rather than
copied, so the live path cannot silently drift from the evaluated one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ptzcalib import data_io  # noqa: E402
from ptzcalib.krt_optimizer import FactorType, KRTOptimizer  # noqa: E402
from ptzcalib.types import Camera  # noqa: E402

from .config import Config  # noqa: E402
from . import geometry  # noqa: E402


def _load_extract_match():
    """Import ``scripts/extract_match.py`` (not a package) as a module."""
    path = os.path.join(_ROOT, "scripts", "extract_match.py")
    spec = importlib.util.spec_from_file_location("ptz_extract_match", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ptz_extract_match"] = module
    spec.loader.exec_module(module)
    return module


def load_offline_quality(ref_params: str) -> dict:
    """Grade that ``eval_own.py`` gave the reference bundle, read from disk.

    This describes the **offline solve the references came from**, not the live
    calibration: absolute accuracy is established once, against surveyed points
    that no live frame can re-observe (they are tile-grid coordinates with no
    physical marker).  The live stage inherits that accuracy and can only be
    watched for *deviation* from it, so the number belongs in the settings card
    next to ``ref_params`` - it is a property of the bundle, like the frame count.

    The report is ``eval_report.json`` beside the parameter JSON, written by
    ``scripts/eval_own.py --holdout`` (EVALUATION.md section 2-6).
    """
    report = os.path.join(os.path.dirname(ref_params), "eval_report.json")
    out = {"report": report, "quality": "unknown"}
    if not os.path.isfile(report):
        out.update(status="missing", reason=f"no eval_report.json beside {os.path.basename(ref_params)}")
        return out
    try:
        with open(report, "r", encoding="utf-8") as fin:
            data = json.load(fin)
    except (OSError, ValueError) as exc:
        out.update(status="error", reason=f"cannot read {os.path.basename(report)}: {exc}")
        return out

    verdict = data.get("verdict")
    if not verdict:
        out.update(
            status="stale",
            reason="report predates the calibration_quality verdict; re-run eval_own.py",
        )
        return out

    out.update(
        status="ok",
        quality=data.get("calibration_quality", "unknown"),
        rmse_m=verdict.get("rmse_m"),
        pass_threshold_m=verdict.get("pass_threshold_m"),
        degraded_threshold_m=verdict.get("degraded_threshold_m"),
        reason=verdict.get("reason"),
        n_obs=verdict.get("n_obs"),
        n_frames_measured=verdict.get("n_frames_measured"),
        registered=verdict.get("registered"),
        total=verdict.get("total"),
        rate=verdict.get("rate"),
    )
    return out


@dataclass
class RefFrame:
    """One registered reference view: features + world camera."""

    name: str
    size: Tuple[int, int]
    features: object  # extract_match.Features
    camera: Camera


class RelocEngine:
    """Holds the reference bundle and the (heavy) networks."""

    def __init__(self, cfg: Config, log=print) -> None:
        self.cfg = cfg
        self.log = log
        self.em = _load_extract_match()

        import torch

        torch.manual_seed(cfg.seed)
        cv2.setRNGSeed(cfg.seed)

        self.device = cfg.resolved_device()
        self.log(f"[engine] device={self.device}")

        self.extractor = self.em.SuperPointExtractor(
            self.device,
            max_keypoints=cfg.max_keypoints,
            keypoint_threshold=cfg.keypoint_threshold,
            nms_radius=cfg.nms_radius,
            resize=cfg.resize,
        )
        self.matcher = self.em.SuperGlueMatcher(self.device, weights=cfg.superglue_weights)

        self.mask: Optional[np.ndarray] = None
        if cfg.mask and os.path.exists(cfg.mask):
            img = cv2.imread(cfg.mask, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise SystemExit(f"cannot read mask: {cfg.mask}")
            self.mask = img > 0
            excluded = self.mask.size - int(self.mask.sum())
            self.log(f"[engine] mask {cfg.mask} ({excluded} px excluded)")

        self.refs: List[RefFrame] = self._load_refs()
        self.survey = _load_survey(cfg.survey_points)
        self.offline_quality = load_offline_quality(cfg.ref_params)
        self.log(
            f"[engine] {len(self.refs)} reference frames, "
            f"{len(self.survey)} survey points, "
            f"factor={'FDist' if cfg.dist else 'F'}"
        )
        oq = self.offline_quality
        if oq["status"] == "ok":
            self.log(
                f"[engine] reference bundle graded {oq['quality']} offline "
                f"(rmse {oq['rmse_m']:.4f} m, n={oq['n_obs']})"
            )
        else:
            self.log(f"[engine] offline grade unavailable: {oq['reason']}")

    # -- reference bundle --------------------------------------------------
    def _load_refs(self) -> List[RefFrame]:
        cfg = self.cfg
        fnames = self.em.list_images(cfg.ref_images)
        if not fnames:
            raise SystemExit(f"no reference images in {cfg.ref_images}")

        ok, cameras = data_io.read_cam_from_json(cfg.ref_params, fnames)
        if not ok:
            raise SystemExit(
                f"{cfg.ref_params} does not cover every image in {cfg.ref_images} "
                "(point --ref_images at the registered-only directory)"
            )

        refs: List[RefFrame] = []
        for i, fname in enumerate(fnames):
            fpath = os.path.join(cfg.ref_features, fname + ".txt")
            if not os.path.exists(fpath):
                raise SystemExit(f"missing reference feature file: {fpath}")
            image = cv2.imread(os.path.join(cfg.ref_images, fname))
            if image is None:
                raise SystemExit(f"cannot read reference image: {fname}")
            size = (image.shape[1], image.shape[0])
            feat = self.em.read_features_txt(fpath, fname, size)
            if feat.desc.shape[1] == 0:
                raise SystemExit(
                    f"{fpath} has no descriptors; re-run the offline matching without --no_desc"
                )
            refs.append(RefFrame(fname, size, feat, cameras[i]))
        return refs

    # -- live frame --------------------------------------------------------
    def extract_live(self, frame_bgr: np.ndarray):
        feat = self.extractor.extract(frame_bgr, "live")
        if self.mask is not None:
            h, w = self.mask.shape
            if feat.size != (w, h):
                raise ValueError(
                    f"live frame {feat.size[0]}x{feat.size[1]} does not match the "
                    f"reference/mask resolution {w}x{h}"
                )
            feat, _dropped = self.em.apply_mask(feat, self.mask)
        return feat

    def calibrate(self, frame_bgr: np.ndarray) -> dict:
        """Full online stage for one frame. Never raises on a normal failure."""
        cfg = self.cfg
        t0 = time.time()
        live = self.extract_live(frame_bgr)
        t_extract = time.time() - t0

        t0 = time.time()
        scored: List[Tuple[int, int, np.ndarray]] = []  # (n_inliers, ref_idx, matches)
        per_ref: List[dict] = []
        for idx, ref in enumerate(self.refs):
            matches, scores = self.matcher.match(ref.features, live)
            inliers, stat = self.em.filter_pair(
                ref.features, live, matches, scores, cfg.conf_thresh, cfg.min_inliers
            )
            per_ref.append({"ref": ref.name, **stat, "kept": int(inliers.shape[0])})
            if inliers.shape[0]:
                scored.append((int(inliers.shape[0]), idx, inliers))
        t_match = time.time() - t0
        per_ref.sort(key=lambda d: -d["inlier"])

        result = {
            "ok": False,
            "num_keypoints": int(live.num),
            "frame_size": [int(live.size[0]), int(live.size[1])],
            "timing_s": {"extract": round(t_extract, 3), "match": round(t_match, 3)},
            "candidates": per_ref[:5],
        }

        if not scored:
            result["reason"] = (
                f"no reference reached {cfg.min_inliers} RANSAC inliers "
                f"(best raw {max((d['raw'] for d in per_ref), default=0)})"
            )
            return result

        scored.sort(key=lambda x: -x[0])
        n_inliers, ref_idx, matches = scored[0]
        ref = self.refs[ref_idx]

        t0 = time.time()
        factor = FactorType.FDist if cfg.dist else FactorType.F
        optimizer = KRTOptimizer(cfg.max_iter, cfg.max_reproj_error, factor)

        # run_ptz_reloc.py: focal seeded from the reference, principal point at
        # the live frame centre, rotation/translation/distortion from the reference.
        f = ref.camera.K[0, 0]
        cx = 0.5 * live.size[0]
        cy = 0.5 * live.size[1]
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        optimizer.set_init_params(K, ref.camera.R, ref.camera.t, ref.camera.dist)
        optimizer.add_2d2d_constraints(
            ref.camera,
            np.asarray(ref.features.kpts, dtype=np.float64),
            np.asarray(live.kpts, dtype=np.float64),
            matches,
        )
        ok, K, R, t, dist = optimizer.solve()
        t_solve = time.time() - t0

        result["timing_s"]["solve"] = round(t_solve, 3)
        result["best_ref"] = ref.name
        result["num_matches"] = int(n_inliers)
        result["num_iterations"] = int(optimizer.num_iter_)

        if not ok:
            result["reason"] = (
                "KRTOptimizer rejected the solve (no convergence, reprojection "
                f">= {cfg.max_reproj_error:g} px, or FOV out of range)"
            )
            return result

        cam = Camera(K, R, t, dist)
        result["ok"] = True
        result["reproj_error_px"] = float(
            optimizer.cal_2d2d_reproj_error(
                ref.camera,
                np.asarray(ref.features.kpts, dtype=np.float64),
                np.asarray(live.kpts, dtype=np.float64),
                matches,
            )
        )
        result["camera"] = camera_to_dict(cam, live.size)
        result["_camera"] = cam
        result["_matches_live_uv"] = np.asarray(live.kpts, dtype=np.float64)[matches[:, 1]]
        return result


def camera_to_dict(cam: Camera, size: Sequence[int]) -> dict:
    """Same field names as the ``*.json`` the offline/online stages write."""
    C = geometry.camera_center(cam)
    fx = float(cam.K[0, 0])
    fy = float(cam.K[1, 1])
    cx = float(cam.K[0, 2])
    cy = float(cam.K[1, 2])
    return {
        "res": [int(size[0]), int(size[1])],
        "K": cam.K.reshape(-1).tolist(),
        "R": cam.R.reshape(-1).tolist(),
        "t": cam.t.reshape(-1).tolist(),
        "dist": cam.dist.reshape(-1).tolist(),
        "pos": C.tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fov_x_deg": float(np.degrees(2.0 * np.arctan(cx / fx))),
        "fov_y_deg": float(np.degrees(2.0 * np.arctan(cy / fy))),
    }


def _load_survey(path: str) -> List[dict]:
    """``survey/points.csv`` -> list of ``{id, X, Y, Z, split}`` (ignores comments)."""
    if not path or not os.path.isfile(path):
        return []
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4 or parts[0].lower() in ("point_id", "id"):
                continue
            try:
                out.append(
                    {
                        "id": parts[0],
                        "xyz": [float(parts[1]), float(parts[2]), float(parts[3])],
                        "split": parts[4] if len(parts) > 4 else "",
                    }
                )
            except ValueError:
                continue
    return out
