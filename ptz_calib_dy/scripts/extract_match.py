#!/usr/bin/env python
"""Phase A - matching pipeline (PLAN.md section 3).

Images -> COLMAP-format feature files + ``pairs_matches.txt``, i.e. exactly the
inputs ``ptzcalib.data_io`` expects.  Nothing under ``ptzcalib/`` is touched:
own data is converted into the public-dataset input format instead.

Run it with the isolated matcher environment::

    .venv-match/bin/python scripts/extract_match.py --mode offline \\
        -i data/infiniq/offline -o data/infiniq/offline_matches

Format notes (PLAN.md section 2-6, verified against ``data_io.py``):

* feature file name is ``<image file name>.txt`` - extension included
* line 1 is ``num_kpts desc_dim``; the following lines are
  ``x y scale orientation desc...``.  ``data_io.read_colmap_features`` drops
  columns 2/3, so the detector score is stashed in the ``scale`` column - that
  lets the online stage re-read reference features (SuperGlue needs the scores)
  without re-running the detector and risking index drift.
* ``pairs_matches.txt`` blocks: ``imgA.jpg imgB.jpg`` header, then
  ``queryIdxA trainIdxB`` lines, blocks separated by a blank line.
* offline writes both directions ``(a,b)`` and ``(b,a)``; online writes only
  ``(reference, test)`` because ``find_best_match`` filters on ``pair[1]``.

Outlier rejection lives here, not in the algorithm: ``load_matches_info``
throws the RANSAC inlier mask away and treats every recorded match as an
inlier, so what is written to disk *is* what the optimiser uses.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SUPERGLUE_DIR = os.path.join(_ROOT, "thirdparty", "superglue")

VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
RANSAC_THRESH = 4.0  # data_io.RANSAC_THRESH
DEFAULT_MIN_INLIERS = 40  # public offline set: smallest recorded pair is exactly 40


# ==========================================================================
# Feature container
# ==========================================================================
@dataclass
class Features:
    """Keypoints in *original image* pixel coordinates."""

    name: str
    size: Tuple[int, int]  # (width, height)
    kpts: np.ndarray  # (N, 2) float32
    scores: np.ndarray  # (N,) float32
    desc: np.ndarray  # (N, D) float32

    @property
    def num(self) -> int:
        return int(self.kpts.shape[0])


class _ShapeOnly:
    """Stand-in for the image tensor: SuperGlue only reads ``.shape``."""

    __slots__ = ("shape",)

    def __init__(self, height: int, width: int) -> None:
        self.shape = (1, 1, height, width)


# ==========================================================================
# Abstract interfaces (PLAN.md section 3, first bullet)
# ==========================================================================
class Extractor:
    """``extract(img) -> Features``"""

    name = "extractor"
    desc_dim = 0

    def extract(self, image_bgr: np.ndarray, name: str) -> Features:
        raise NotImplementedError


class Matcher:
    """``match(f1, f2) -> (matches (M,2) int32, scores (M,) float32)``"""

    name = "matcher"

    def match(self, f0: Features, f1: Features) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# ==========================================================================
# SuperPoint / SuperGlue (Magic Leap) - noncommercial research use only
# ==========================================================================
class SuperPointExtractor(Extractor):
    name = "superpoint"
    desc_dim = 256

    def __init__(
        self,
        device: str,
        max_keypoints: int = 2048,
        keypoint_threshold: float = 0.005,
        nms_radius: int = 4,
        resize: int = 1024,
    ) -> None:
        if _SUPERGLUE_DIR not in sys.path:
            sys.path.insert(0, _SUPERGLUE_DIR)
        from models.superpoint import SuperPoint  # noqa: E402  (vendored)

        self.device = device
        self.resize = resize
        self.net = (
            SuperPoint(
                {
                    "max_keypoints": max_keypoints,
                    "keypoint_threshold": keypoint_threshold,
                    "nms_radius": nms_radius,
                }
            )
            .eval()
            .to(device)
        )

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray, name: str) -> Features:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if self.resize > 0 and max(h, w) != self.resize:
            s = self.resize / float(max(h, w))
            nw, nh = int(round(w * s)), int(round(h * s))
            small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            nw, nh = w, h
            small = gray
        # same convention as SuperGluePretrainedNetwork/models/utils.py:read_image
        scales = np.array([w / float(nw), h / float(nh)], dtype=np.float64)

        inp = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(self.device)
        pred = self.net({"image": inp})

        kpts = pred["keypoints"][0].detach().cpu().numpy().astype(np.float64) * scales
        scores = pred["scores"][0].detach().cpu().numpy().astype(np.float32)
        desc = pred["descriptors"][0].detach().cpu().numpy().T.astype(np.float32)  # (N, 256)
        return Features(name, (w, h), kpts.astype(np.float32), scores, desc)


class SuperGlueMatcher(Matcher):
    name = "superglue"

    def __init__(
        self,
        device: str,
        weights: str = "outdoor",
        sinkhorn_iterations: int = 20,
        match_threshold: float = 0.2,
    ) -> None:
        if _SUPERGLUE_DIR not in sys.path:
            sys.path.insert(0, _SUPERGLUE_DIR)
        from models.superglue import SuperGlue  # noqa: E402  (vendored)

        self.device = device
        self.net = (
            SuperGlue(
                {
                    "weights": weights,
                    "sinkhorn_iterations": sinkhorn_iterations,
                    "match_threshold": match_threshold,
                }
            )
            .eval()
            .to(device)
        )

    @torch.no_grad()
    def match(self, f0: Features, f1: Features) -> Tuple[np.ndarray, np.ndarray]:
        if f0.num == 0 or f1.num == 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        def _t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).float()[None].to(self.device)

        data = {
            "keypoints0": _t(f0.kpts),
            "keypoints1": _t(f1.kpts),
            "scores0": _t(f0.scores),
            "scores1": _t(f1.scores),
            "descriptors0": _t(f0.desc.T),  # (1, D, N)
            "descriptors1": _t(f1.desc.T),
            "image0": _ShapeOnly(f0.size[1], f0.size[0]),
            "image1": _ShapeOnly(f1.size[1], f1.size[0]),
        }
        pred = self.net(data)
        m0 = pred["matches0"][0].detach().cpu().numpy()
        sc = pred["matching_scores0"][0].detach().cpu().numpy()

        valid = m0 > -1
        idx0 = np.nonzero(valid)[0]
        matches = np.stack([idx0, m0[valid]], axis=1).astype(np.int32)
        return matches, sc[valid].astype(np.float32)


# ==========================================================================
# Apache-2.0 / BSD alternatives (PLAN.md section 2-5: swap when the licence
# matters).  kornia ships the weights; keypoints stay in original pixels.
# ==========================================================================
class KorniaExtractor(Extractor):
    def __init__(self, kind: str, device: str, max_keypoints: int = 2048, resize: int = 1024) -> None:
        import kornia.feature as KF

        self.kind = kind
        self.name = kind
        self.device = device
        self.resize = resize
        self.max_keypoints = max_keypoints
        if kind == "disk":
            self.net = KF.DISK.from_pretrained("depth").to(device).eval()
            self.desc_dim = 128
        elif kind == "aliked":
            self.net = KF.ALIKED(model_name="aliked-n16rot", max_num_keypoints=max_keypoints).to(device).eval()
            self.desc_dim = 128
        else:
            raise ValueError(f"unknown kornia extractor: {kind}")

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray, name: str) -> Features:
        h, w = image_bgr.shape[:2]
        if self.resize > 0 and max(h, w) != self.resize:
            s = self.resize / float(max(h, w))
            nw, nh = int(round(w * s)), int(round(h * s))
            small = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            nw, nh = w, h
            small = image_bgr
        scales = np.array([w / float(nw), h / float(nh)], dtype=np.float64)

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device)

        if self.kind == "disk":
            feats = self.net(inp, n=self.max_keypoints, window_size=5, pad_if_not_divisible=True)[0]
            kpts = feats.keypoints.detach().cpu().numpy().astype(np.float64)
            desc = feats.descriptors.detach().cpu().numpy().astype(np.float32)
            scores = feats.detection_scores.detach().cpu().numpy().astype(np.float32)
        else:
            out = self.net(inp)
            kpts = out["keypoints"][0].detach().cpu().numpy().astype(np.float64)
            desc = out["descriptors"][0].detach().cpu().numpy().astype(np.float32)
            scores = out["scores"][0].detach().cpu().numpy().astype(np.float32)

        return Features(name, (w, h), (kpts * scales).astype(np.float32), scores, desc)


class LightGlueMatcher(Matcher):
    def __init__(self, feature_name: str, device: str) -> None:
        import kornia.feature as KF

        self.name = f"lightglue({feature_name})"
        self.device = device
        self.net = KF.LightGlue(feature_name).to(device).eval()

    @torch.no_grad()
    def match(self, f0: Features, f1: Features) -> Tuple[np.ndarray, np.ndarray]:
        if f0.num == 0 or f1.num == 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        def _pack(f: Features) -> dict:
            return {
                "keypoints": torch.from_numpy(f.kpts).float()[None].to(self.device),
                "descriptors": torch.from_numpy(f.desc).float()[None].to(self.device),
                "image_size": torch.tensor([[f.size[0], f.size[1]]], dtype=torch.float32, device=self.device),
            }

        out = self.net({"image0": _pack(f0), "image1": _pack(f1)})
        matches = out["matches"][0].detach().cpu().numpy().astype(np.int32)
        scores = out["scores"][0].detach().cpu().numpy().astype(np.float32)
        return matches.reshape(-1, 2), scores.reshape(-1)


def build_extractor_matcher(args: argparse.Namespace) -> Tuple[Extractor, Matcher]:
    if args.extractor == "superpoint":
        ex: Extractor = SuperPointExtractor(
            args.device,
            max_keypoints=args.max_keypoints,
            keypoint_threshold=args.keypoint_threshold,
            nms_radius=args.nms_radius,
            resize=args.resize,
        )
    else:
        ex = KorniaExtractor(args.extractor, args.device, max_keypoints=args.max_keypoints, resize=args.resize)

    if args.matcher == "superglue":
        if args.extractor != "superpoint":
            raise SystemExit("superglue only accepts superpoint features")
        mt: Matcher = SuperGlueMatcher(
            args.device, weights=args.superglue_weights, sinkhorn_iterations=args.sinkhorn_iterations
        )
    else:
        mt = LightGlueMatcher(args.extractor, args.device)
    return ex, mt


# ==========================================================================
# COLMAP-format I/O
# ==========================================================================
def list_images(img_dir: str) -> List[str]:
    """Mirror ``data_io.load_imgs_and_features``: sorted, mask.png excluded."""
    out = []
    for fname in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_IMG_EXTS or fname == "mask.png":
            continue
        out.append(fname)
    return out


def write_features_txt(path: str, feat: Features, write_desc: bool) -> None:
    n = feat.num
    dim = feat.desc.shape[1] if (write_desc and n) else 0
    cols = [feat.kpts[:, 0], feat.kpts[:, 1], feat.scores, np.zeros(n, dtype=np.float32)]
    body = np.stack(cols, axis=1)
    if dim:
        body = np.concatenate([body, feat.desc], axis=1)
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(f"{n} {dim}\n")
        np.savetxt(fout, body, fmt="%.6g")


def read_features_txt(path: str, name: str, size: Tuple[int, int]) -> Features:
    """Inverse of :func:`write_features_txt` (score recovered from col 2)."""
    with open(path, "r", encoding="utf-8") as fin:
        num, dim = (int(x) for x in fin.readline().split()[:2])
        values = np.fromstring(fin.read(), dtype=np.float32, sep=" ")
    stride = 4 + dim
    values = values[: num * stride].reshape(num, stride)
    return Features(name, size, values[:, 0:2], values[:, 2].copy(), values[:, 4:].copy())


def write_pairs_matches(path: str, blocks: Sequence[Tuple[str, str, np.ndarray]]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        for name_a, name_b, matches in blocks:
            fout.write(f"{name_a} {name_b}\n")
            for q, t in matches:
                fout.write(f"{int(q)} {int(t)}\n")
            fout.write("\n")


# ==========================================================================
# Pair matching + outlier filtering
# ==========================================================================
def filter_pair(
    f0: Features,
    f1: Features,
    matches: np.ndarray,
    scores: np.ndarray,
    conf_thresh: float,
    min_inliers: int,
) -> Tuple[np.ndarray, dict]:
    """Confidence gate -> RANSAC homography -> inliers only."""
    stat = {"raw": int(matches.shape[0]), "conf": 0, "inlier": 0}

    if matches.shape[0]:
        keep = scores >= conf_thresh
        matches = matches[keep]
    stat["conf"] = int(matches.shape[0])

    if matches.shape[0] < 4:
        return np.zeros((0, 2), dtype=np.int32), stat

    pts0 = f0.kpts[matches[:, 0]].astype(np.float32)
    pts1 = f1.kpts[matches[:, 1]].astype(np.float32)
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, RANSAC_THRESH)
    if H is None or mask is None:
        return np.zeros((0, 2), dtype=np.int32), stat

    inliers = matches[mask.ravel().astype(bool)]
    stat["inlier"] = int(inliers.shape[0])
    if inliers.shape[0] < min_inliers:
        return np.zeros((0, 2), dtype=np.int32), stat
    return inliers.astype(np.int32), stat


def near_identity_ratio(
    feats: Dict[str, Features], blocks: Sequence[Tuple[str, str, np.ndarray]], tol: float = 1.5
) -> dict:
    """Share of matches with (almost) zero disparity.

    A burned-in OSD (timestamp, channel name, PTZ readout) sits at the same
    pixel in every frame, so it matches itself perfectly.  Those matches are
    mutually consistent with an identity homography, which means RANSAC keeps
    them - and they tell the bundle adjustment the camera never rotated.  A
    non-trivial ratio here means the overlay must be masked out.
    """
    total = 0
    ident = 0
    ys: List[float] = []
    for name_a, name_b, matches in blocks:
        if not matches.shape[0]:
            continue
        pa = feats[name_a].kpts[matches[:, 0]]
        pb = feats[name_b].kpts[matches[:, 1]]
        sel = np.linalg.norm(pa - pb, axis=1) < tol
        total += int(matches.shape[0])
        ident += int(sel.sum())
        ys.extend(pa[sel, 1].tolist())
    out = {"total": total, "near_identity": ident, "ratio": (ident / total) if total else 0.0}
    if ys:
        a = np.asarray(ys)
        out["y_range"] = [float(np.percentile(a, 1)), float(np.percentile(a, 99))]
    return out


# ==========================================================================
# Keypoint masking (the algorithm has no masking; it must happen here)
# ==========================================================================
def build_mask(mask_path: str, rects: Sequence[str], size: Tuple[int, int]) -> Optional[np.ndarray]:
    """``True`` where keypoints are kept.  ``rects`` are ``x0,y0,x1,y1`` strings."""
    w, h = size
    mask: Optional[np.ndarray] = None

    if mask_path:
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"cannot read mask: {mask_path}")
        if (img.shape[1], img.shape[0]) != (w, h):
            raise SystemExit(f"mask size {img.shape[1]}x{img.shape[0]} != image size {w}x{h}")
        mask = img > 0

    for rect in rects:
        try:
            x0, y0, x1, y1 = (int(v) for v in rect.split(","))
        except ValueError:
            raise SystemExit(f"--exclude_rect expects x0,y0,x1,y1 (got {rect!r})")
        if mask is None:
            mask = np.ones((h, w), dtype=bool)
        mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = False

    return mask


def apply_mask(feat: Features, mask: np.ndarray) -> Tuple[Features, int]:
    if feat.num == 0:
        return feat, 0
    h, w = mask.shape
    xi = np.clip(np.round(feat.kpts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(feat.kpts[:, 1]).astype(int), 0, h - 1)
    keep = mask[yi, xi]
    dropped = int((~keep).sum())
    return (
        Features(feat.name, feat.size, feat.kpts[keep], feat.scores[keep], feat.desc[keep]),
        dropped,
    )


def draw_pair(img0: np.ndarray, img1: np.ndarray, f0: Features, f1: Features, matches: np.ndarray) -> np.ndarray:
    h = max(img0.shape[0], img1.shape[0])
    canvas = np.zeros((h, img0.shape[1] + img1.shape[1], 3), dtype=np.uint8)
    canvas[: img0.shape[0], : img0.shape[1]] = img0
    canvas[: img1.shape[0], img0.shape[1] :] = img1
    off = np.array([img0.shape[1], 0], dtype=np.float32)

    step = max(1, matches.shape[0] // 200)
    rng = np.random.default_rng(0)
    for q, t in matches[::step]:
        p0 = tuple(np.round(f0.kpts[q]).astype(int))
        p1 = tuple(np.round(f1.kpts[t] + off).astype(int))
        color = tuple(int(c) for c in rng.integers(64, 255, 3))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    return canvas


# ==========================================================================
# Track statistics (PLAN.md section 3 last bullet / CP-C gate)
# ==========================================================================
def track_stats(num_images: int, blocks: Sequence[Tuple[int, int, np.ndarray]], min_len: int = 4) -> dict:
    """Reuse the real ``TracksBuilder`` so the numbers match what PTZ-IBA sees."""
    from ptzcalib.tracks import TracksBuilder, length
    from ptzcalib.types import MatchesInfo

    matches_info = []
    for i, j, matches in blocks:
        info = MatchesInfo()
        info.src_img_idx = i
        info.dst_img_idx = j
        info.matches = matches
        matches_info.append(info)

    builder = TracksBuilder()
    builder.build(matches_info)
    builder.filter(min_len)
    tracks = builder.export_to_stl()

    lengths = np.array([len(t) for t in tracks.values()], dtype=np.int32)
    total, max_len, min_len_seen = length(tracks)
    hist: Dict[str, int] = {}
    for n in np.unique(lengths):
        hist[str(int(n))] = int((lengths == n).sum())
    return {
        "num_tracks": int(lengths.size),
        "num_tracks_ge4": int((lengths >= 4).sum()),
        "mean_length": float(lengths.mean()) if lengths.size else 0.0,
        "max_length": int(max_len),
        "min_length": int(min_len_seen) if lengths.size else 0,
        "total_observations": int(total),
        "length_histogram": hist,
    }


def _dist_summary(values: Sequence[int]) -> dict:
    if not len(values):
        return {"count": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "min": int(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": int(a.max()),
        "mean": float(a.mean()),
        "total": int(a.sum()),
    }


# ==========================================================================
# PTZ readout pre-filter (online)
# ==========================================================================
def load_ptz_log(path: str) -> Dict[str, Tuple[float, float, float]]:
    log: Dict[str, Tuple[float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = [p.strip() for p in line.replace(",", " ").split()]
            if len(parts) < 4 or not parts[0] or parts[0].lower() == "fname":
                continue
            log[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return log


def angular_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dpan = abs(a[0] - b[0]) % 360.0
    dpan = min(dpan, 360.0 - dpan)
    return math.hypot(dpan, a[1] - b[1])


# ==========================================================================
# Modes
# ==========================================================================
def _extract_all(
    extractor: Extractor, img_dir: str, fnames: Sequence[str], args: argparse.Namespace
) -> Tuple[Dict[str, Features], int]:
    feats: Dict[str, Features] = {}
    mask: Optional[np.ndarray] = None
    mask_path = args.mask
    if not mask_path and not args.no_auto_mask:
        auto = os.path.join(img_dir, "mask.png")  # data_io skips mask.png when loading images
        if os.path.exists(auto):
            mask_path = auto
    total_masked = 0

    for fname in fnames:
        image = cv2.imread(os.path.join(img_dir, fname))
        if image is None or image.size == 0:
            raise SystemExit(f"cannot read image: {fname}")
        f = extractor.extract(image, fname)

        if mask_path or args.exclude_rect:
            if mask is None:
                mask = build_mask(mask_path, args.exclude_rect, f.size)
                print(f"  [mask] {mask.size - int(mask.sum())} px excluded ({mask_path or 'rects only'})")
            f, dropped = apply_mask(f, mask)
            total_masked += dropped

        feats[fname] = f
        if args.verbose:
            print(f"  [extract] {fname}  {f.size[0]}x{f.size[1]}  {f.num} kpts")
    return feats, total_masked


def run_offline(args: argparse.Namespace) -> dict:
    extractor, matcher = build_extractor_matcher(args)
    fnames = list_images(args.images)
    if len(fnames) < 2:
        raise SystemExit(f"need at least 2 images, found {len(fnames)} in {args.images}")

    os.makedirs(args.output, exist_ok=True)
    print(f"[Phase A/offline] {len(fnames)} images  {extractor.name}+{matcher.name}  device={args.device}")

    t0 = time.time()
    feats, num_masked = _extract_all(extractor, args.images, fnames, args)
    t_extract = time.time() - t0

    for fname in fnames:
        write_features_txt(os.path.join(args.output, fname + ".txt"), feats[fname], not args.no_desc)
    print(f"[Phase A/offline] features written ({t_extract:.1f}s)")

    t0 = time.time()
    kept: List[Tuple[int, int, np.ndarray]] = []
    dropped: List[dict] = []
    raw_counts: List[int] = []
    for i, j in itertools.combinations(range(len(fnames)), 2):
        f0, f1 = feats[fnames[i]], feats[fnames[j]]
        matches, scores = matcher.match(f0, f1)
        inliers, stat = filter_pair(f0, f1, matches, scores, args.conf_thresh, args.min_inliers)
        raw_counts.append(stat["raw"])
        if inliers.shape[0]:
            kept.append((i, j, inliers))
        else:
            dropped.append({"pair": [fnames[i], fnames[j]], **stat})
        if args.verbose:
            print(
                f"  [match] {fnames[i]} <-> {fnames[j]}: raw={stat['raw']} "
                f"conf={stat['conf']} inlier={stat['inlier']}"
                f"{'' if inliers.shape[0] else '  (dropped)'}"
            )
    t_match = time.time() - t0

    # bidirectional blocks - load_matches_info fills only [i*n+j]
    blocks: List[Tuple[str, str, np.ndarray]] = []
    directed: List[Tuple[int, int, np.ndarray]] = []
    for i, j, m in kept:
        blocks.append((fnames[i], fnames[j], m))
        directed.append((i, j, m))
        blocks.append((fnames[j], fnames[i], m[:, ::-1].copy()))
        directed.append((j, i, m[:, ::-1].copy()))
    write_pairs_matches(os.path.join(args.output, "pairs_matches.txt"), blocks)

    num_possible = len(fnames) * (len(fnames) - 1) // 2
    print(
        f"[Phase A/offline] {len(kept)}/{num_possible} pairs kept "
        f"({len(blocks)} directed blocks, {t_match:.1f}s)"
    )

    if args.viz:
        viz_dir = os.path.join(args.output, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        for i, j, m in sorted(kept, key=lambda x: -x[2].shape[0])[: args.viz]:
            img0 = cv2.imread(os.path.join(args.images, fnames[i]))
            img1 = cv2.imread(os.path.join(args.images, fnames[j]))
            out = draw_pair(img0, img1, feats[fnames[i]], feats[fnames[j]], m)
            stem = f"{os.path.splitext(fnames[i])[0]}__{os.path.splitext(fnames[j])[0]}_{m.shape[0]}.jpg"
            cv2.imwrite(os.path.join(viz_dir, stem), out)
        for i, j, m in sorted(kept, key=lambda x: x[2].shape[0])[: args.viz]:
            img0 = cv2.imread(os.path.join(args.images, fnames[i]))
            img1 = cv2.imread(os.path.join(args.images, fnames[j]))
            out = draw_pair(img0, img1, feats[fnames[i]], feats[fnames[j]], m)
            stem = f"{os.path.splitext(fnames[i])[0]}__{os.path.splitext(fnames[j])[0]}_{m.shape[0]}.jpg"
            cv2.imwrite(os.path.join(viz_dir, stem), out)
        print(f"[Phase A/offline] wrote match visualisations to {viz_dir}")

    per_image = {fnames[i]: 0 for i in range(len(fnames))}
    for i, j, _ in kept:
        per_image[fnames[i]] += 1
        per_image[fnames[j]] += 1

    report = {
        "mode": "offline",
        "images_dir": os.path.abspath(args.images),
        "output_dir": os.path.abspath(args.output),
        "extractor": extractor.name,
        "matcher": matcher.name,
        "config": {
            "resize": args.resize,
            "max_keypoints": args.max_keypoints,
            "keypoint_threshold": args.keypoint_threshold,
            "conf_thresh": args.conf_thresh,
            "min_inliers": args.min_inliers,
            "ransac_thresh": RANSAC_THRESH,
        },
        "num_images": len(fnames),
        "num_keypoints": _dist_summary([feats[f].num for f in fnames]),
        "keypoints_masked_out": num_masked,
        "near_identity_matches": near_identity_ratio(feats, blocks),
        "pairs_possible": num_possible,
        "pairs_kept": len(kept),
        "directed_blocks": len(blocks),
        "matches_raw_per_pair": _dist_summary(raw_counts),
        "matches_written_per_pair": _dist_summary([m.shape[0] for _, _, m in kept]),
        "pairs_per_image": _dist_summary(list(per_image.values())),
        "pairs_per_image_detail": per_image,
        "isolated_images": [n for n, c in per_image.items() if c == 0],
        "dropped_pairs": dropped if args.verbose else len(dropped),
        "tracks": track_stats(len(fnames), directed, min_len=4),
        "timing_s": {"extract": round(t_extract, 2), "match": round(t_match, 2)},
    }
    return report


def run_online(args: argparse.Namespace) -> dict:
    extractor, matcher = build_extractor_matcher(args)
    ref_fnames = list_images(args.ref_images)
    test_fnames = list_images(args.test_images)
    if not ref_fnames or not test_fnames:
        raise SystemExit("reference or test image directory is empty")

    os.makedirs(args.output, exist_ok=True)
    print(
        f"[Phase A/online] {len(test_fnames)} test vs {len(ref_fnames)} ref  "
        f"{extractor.name}+{matcher.name}  device={args.device}"
    )

    # Reference features are re-read from the offline run so the feature indices
    # written here refer to exactly the same keypoints run_ptz_reloc.py loads.
    ref_feats: Dict[str, Features] = {}
    for fname in ref_fnames:
        fpath = os.path.join(args.ref_features, fname + ".txt")
        if not os.path.exists(fpath):
            raise SystemExit(f"missing reference feature file: {fpath}")
        image = cv2.imread(os.path.join(args.ref_images, fname))
        f = read_features_txt(fpath, fname, (image.shape[1], image.shape[0]))
        if f.desc.shape[1] == 0:
            raise SystemExit(
                f"{fpath} has no descriptors; re-run the offline stage without --no_desc"
            )
        ref_feats[fname] = f

    t0 = time.time()
    test_feats, num_masked = _extract_all(extractor, args.test_images, test_fnames, args)
    t_extract = time.time() - t0
    for fname in test_fnames:
        write_features_txt(os.path.join(args.output, fname + ".txt"), test_feats[fname], not args.no_desc)

    ptz_log = load_ptz_log(args.ptz_log) if args.ptz_log else {}
    if ptz_log:
        missing_ref = [r for r in ref_fnames if r not in ptz_log]
        missing_test = [t for t in test_fnames if t not in ptz_log]
        if missing_ref:
            print(
                f"  [warn] {len(missing_ref)} reference frames are absent from {args.ptz_log} and will "
                f"never be pre-selected, e.g. {missing_ref[:3]}"
            )
        if missing_test:
            print(
                f"  [warn] {len(missing_test)} test frames are absent from {args.ptz_log}; "
                f"they fall back to matching against every reference"
            )

    t0 = time.time()
    blocks: List[Tuple[str, str, np.ndarray]] = []
    per_test: Dict[str, dict] = {}
    for tname in test_fnames:
        candidates = list(ref_fnames)
        if ptz_log and tname in ptz_log:
            known = [r for r in candidates if r in ptz_log]
            if known:
                known.sort(key=lambda r: angular_distance(ptz_log[r], ptz_log[tname]))
                candidates = known[: args.prefilter]

        scored: List[Tuple[int, str, np.ndarray]] = []
        for rname in candidates:
            f_ref, f_test = ref_feats[rname], test_feats[tname]
            matches, scores = matcher.match(f_ref, f_test)
            inliers, _stat = filter_pair(f_ref, f_test, matches, scores, args.conf_thresh, args.min_inliers)
            if inliers.shape[0]:
                scored.append((int(inliers.shape[0]), rname, inliers))

        scored.sort(key=lambda x: -x[0])
        top = scored[: args.topk]
        for n, rname, inliers in top:
            # (reference, test) direction: find_best_match filters on pair[1]
            blocks.append((rname, tname, inliers))
        per_test[tname] = {
            "candidates": len(candidates),
            "refs_written": [r for _, r, _ in top],
            "matches": [n for n, _, _ in top],
        }
        if args.verbose:
            print(f"  [online] {tname}: {per_test[tname]}")
    t_match = time.time() - t0

    write_pairs_matches(os.path.join(args.output, "pairs_matches.txt"), blocks)
    unmatched = [t for t, v in per_test.items() if not v["refs_written"]]
    print(
        f"[Phase A/online] {len(test_fnames) - len(unmatched)}/{len(test_fnames)} test frames "
        f"matched, {len(blocks)} blocks ({t_match:.1f}s)"
    )

    best_counts = [v["matches"][0] for v in per_test.values() if v["matches"]]
    return {
        "mode": "online",
        "ref_images_dir": os.path.abspath(args.ref_images),
        "test_images_dir": os.path.abspath(args.test_images),
        "output_dir": os.path.abspath(args.output),
        "extractor": extractor.name,
        "matcher": matcher.name,
        "config": {
            "resize": args.resize,
            "max_keypoints": args.max_keypoints,
            "conf_thresh": args.conf_thresh,
            "min_inliers": args.min_inliers,
            "topk": args.topk,
            "prefilter": args.prefilter,
            "ptz_log": args.ptz_log or None,
        },
        "num_ref": len(ref_fnames),
        "num_test": len(test_fnames),
        "keypoints_masked_out": num_masked,
        "num_blocks": len(blocks),
        "matched_test_frames": len(test_fnames) - len(unmatched),
        "unmatched_test_frames": unmatched,
        "best_match_count": _dist_summary(best_counts),
        "per_test": per_test,
        "timing_s": {"extract": round(t_extract, 2), "match": round(t_match, 2)},
    }


# ==========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTZ-Calib Phase A: features + matches")
    p.add_argument("--mode", choices=["offline", "online"], default="offline")
    p.add_argument("-i", "--images", help="offline: reference images directory")
    p.add_argument("--ref_images", help="online: reference images directory")
    p.add_argument("--ref_features", help="online: offline_matches directory (reference features)")
    p.add_argument("--test_images", help="online: test images directory")
    p.add_argument("-o", "--output", required=True, help="output features/matches directory")

    p.add_argument("--extractor", choices=["superpoint", "disk", "aliked"], default="superpoint")
    p.add_argument("--matcher", choices=["superglue", "lightglue"], default="superglue")
    p.add_argument("--superglue_weights", choices=["outdoor", "indoor"], default="outdoor")
    p.add_argument("--sinkhorn_iterations", type=int, default=20)
    p.add_argument("--max_keypoints", type=int, default=2048)
    # 0.01 rather than SuperPoint's own 0.005: with 0.005 the scene_01 round-trip
    # picked a seed pair whose two-view solve collapsed (focal -> 0.006, 2/30
    # registered).  PTZ-IBA's initial-pair choice is sensitive to which pair ends
    # up with the most matches, so changing detector settings can flip it.
    p.add_argument("--keypoint_threshold", type=float, default=0.01)
    p.add_argument("--nms_radius", type=int, default=4)
    p.add_argument("--resize", type=int, default=-1, help="longest side for detection; -1 keeps original")

    p.add_argument("--mask", default="", help="mask image, non-zero = keep (default: <images>/mask.png)")
    p.add_argument("--no_auto_mask", action="store_true", help="ignore <images>/mask.png")
    p.add_argument(
        "--exclude_rect",
        action="append",
        default=[],
        metavar="x0,y0,x1,y1",
        help="drop keypoints inside this rectangle (repeatable) - e.g. a burned-in OSD banner",
    )

    p.add_argument("--conf_thresh", type=float, default=0.5, help="matcher confidence gate")
    p.add_argument("--min_inliers", type=int, default=DEFAULT_MIN_INLIERS)
    p.add_argument("--topk", type=int, default=3, help="online: references written per test frame")
    p.add_argument("--prefilter", type=int, default=8, help="online: candidates kept by PTZ readout angle")
    p.add_argument("--ptz_log", default="", help="online: CSV fname,pan,tilt,zoom")

    p.add_argument("--no_desc", action="store_true", help="write desc_dim=0 (smaller files, blocks online mode)")
    p.add_argument("--viz", type=int, default=0, help="offline: save N best + N worst pair visualisations")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", default="", help="report path (default: <output>/match_report.json)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    cv2.setRNGSeed(args.seed)

    if args.mode == "offline":
        if not args.images:
            raise SystemExit("--mode offline requires -i/--images")
        report = run_offline(args)
    else:
        missing = [k for k in ("ref_images", "ref_features", "test_images") if not getattr(args, k)]
        if missing:
            raise SystemExit(f"--mode online requires: {', '.join('--' + m for m in missing)}")
        report = run_online(args)

    report_path = args.report or os.path.join(args.output, "match_report.json")
    with open(report_path, "w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2)
        fout.write("\n")

    print("--------------------------------------------------------------")
    if report["mode"] == "offline":
        tr = report["tracks"]
        mw = report["matches_written_per_pair"]
        print(f"keypoints/image : {report['num_keypoints']['median']:.0f} (median)")
        print(f"pairs kept      : {report['pairs_kept']}/{report['pairs_possible']}")
        print(
            f"matches/pair    : min {mw.get('min', 0)}  median {mw.get('median', 0):.0f}  "
            f"max {mw.get('max', 0)}"
        )
        print(f"pairs/image     : min {report['pairs_per_image']['min']}  median {report['pairs_per_image']['median']:.0f}")
        print(f"tracks (len>=4) : {tr['num_tracks_ge4']}   [CP-C gate: >= 1000]")
        print(f"track length    : mean {tr['mean_length']:.2f}  max {tr['max_length']}")
        ni = report["near_identity_matches"]
        flag = "  <-- static overlay? mask it" if ni["ratio"] > 0.02 else ""
        print(f"zero-disparity  : {ni['near_identity']}/{ni['total']} ({100 * ni['ratio']:.1f}%){flag}")
        if report["keypoints_masked_out"]:
            print(f"masked keypoints: {report['keypoints_masked_out']}")
        if report["isolated_images"]:
            print(f"isolated images : {report['isolated_images']}")
    else:
        bm = report["best_match_count"]
        print(f"matched frames  : {report['matched_test_frames']}/{report['num_test']}")
        if bm.get("count"):
            print(f"best matches    : min {bm['min']}  median {bm['median']:.0f}  max {bm['max']}")
        if report["unmatched_test_frames"]:
            print(f"unmatched       : {report['unmatched_test_frames']}")
    print(f"report          : {report_path}")
    print("--------------------------------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
