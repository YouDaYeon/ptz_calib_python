"""Port of ``src/core/data_io.{h,cc}``."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from .logging_utils import PLOGD, PLOGE, PLOGI
from .os_path import basename, listdir, mkdir_ifnot_exist, splitext
from .types import Camera, ImageFeatures, MatchesInfo

VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
RANSAC_THRESH = 4.0
MAX_NUM_MATCHES = 100


# --------------------------------------------------------------------------
# Colmap-format readers
# --------------------------------------------------------------------------
def read_colmap_features(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """``ReadColmapFeatures``: returns ``(keypoints (N,2), descriptors (N,D))``."""
    try:
        with open(filepath, "r", encoding="utf-8") as fin:
            header = fin.readline()
            if not header:
                return np.zeros((0, 2)), None
            num_kpts, desc_dim = (int(x) for x in header.split()[:2])
            body = fin.read()

        values = np.fromstring(body, dtype=np.float64, sep=" ")
        stride = 4 + desc_dim
        expected = num_kpts * stride
        if values.size < expected:
            PLOGD(f"Cannot read colmap features from: {filepath}")
            return np.zeros((0, 2)), None
        values = values[:expected].reshape(num_kpts, stride)

        # C++ stores x/y in a cv::KeyPoint (float) and drops scale/orientation
        kpts = values[:, 0:2].astype(np.float32).astype(np.float64)
        desc = values[:, 4:].astype(np.float32)
        return kpts, desc
    except (OSError, ValueError):
        PLOGD(f"Cannot read colmap features from: {filepath}")
        return np.zeros((0, 2)), None


def _has_image_ending(name: str) -> bool:
    return name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")


def read_colmap_matches(filepath: str) -> Tuple[List[np.ndarray], List[Tuple[str, str]]]:
    """``ReadColmapMatches``: blocks separated by blank lines, ``img1 img2`` header."""
    pairs_matches: List[np.ndarray] = []
    img_pairs_name: List[Tuple[str, str]] = []

    matches: List[Tuple[int, int]] = []
    img_pair: Tuple[str, str] = ("", "")

    try:
        with open(filepath, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    if matches:
                        pairs_matches.append(np.asarray(matches, dtype=np.int32))
                        img_pairs_name.append(img_pair)
                        matches = []
                        img_pair = ("", "")
                    continue

                parts = line.split()
                if not parts:
                    continue
                str1 = parts[0]
                str2 = parts[1] if len(parts) > 1 else ""

                if _has_image_ending(str1):
                    img_pair = (str1, str2)
                else:
                    matches.append((int(str1), int(str2)))

        if matches:  # last block without a trailing blank line
            pairs_matches.append(np.asarray(matches, dtype=np.int32))
            img_pairs_name.append(img_pair)
    except OSError:
        PLOGD(f"Cannot read colmap matches from: {filepath}")

    return pairs_matches, img_pairs_name


# --------------------------------------------------------------------------
# JSON I/O
# --------------------------------------------------------------------------
def _memcpy_into(default: np.ndarray, values) -> np.ndarray:
    """Mimic ``memcpy(mat.data, vec.data(), vec.size() * sizeof(double))``.

    The WorldCup14 annotation files store empty ``K``/``R``/``t``/``dist`` arrays;
    the C++ reader copies zero bytes and silently keeps the default-constructed
    camera, so the port must do the same instead of raising.
    """
    out = default.copy()
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = min(flat.size, out.size)
    if n:
        out.reshape(-1)[:n] = flat[:n]
    return out


def _read_json_file(filepath: str):
    if not filepath:
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as fin:
            return json.load(fin)
    except (OSError, ValueError):
        return {}


def save_to_json(
    cameras: Sequence[Camera],
    names: Sequence[str],
    pixels_gt: Sequence[np.ndarray],
    pts3d_gt: Sequence[np.ndarray],
    filepath: str,
) -> bool:
    """``SaveToJson``: keeps the exact key order / schema of the C++ writer."""
    j_cameras: "OrderedDict[str, dict]" = OrderedDict()

    for i, cam in enumerate(cameras):
        rootname, _ = splitext(names[i])

        j: "OrderedDict[str, object]" = OrderedDict()
        j["name"] = rootname
        t_wc = cam.t_wc()
        j["pos"] = [float(t_wc[0, 0]), float(t_wc[1, 0]), float(t_wc[2, 0])]
        width = int(2 * cam.K[0, 2])
        height = int(2 * cam.K[1, 2])
        j["res"] = [width, height]
        j["K"] = [float(x) for x in cam.K.reshape(-1)]
        j["R"] = [float(x) for x in cam.R.reshape(-1)]
        j["t"] = [float(x) for x in cam.t.reshape(-1)]
        dist_vec = [float(x) for x in cam.dist.reshape(-1)]
        j["dist"] = dist_vec
        j["distType"] = "" if dist_vec[0] < 1e-5 else "k1"

        pix: List[List[float]] = []
        pos: List[List[float]] = []
        pixels_i = np.asarray(pixels_gt[i], dtype=np.float64).reshape(-1, 2)
        pts3d_i = np.asarray(pts3d_gt[i], dtype=np.float64).reshape(-1, 3)
        for k in range(pixels_i.shape[0]):
            # C++ stores the normalised pixels in a vector<vector<float>>
            pix.append(
                [
                    float(np.float32(np.float32(pixels_i[k, 0]) / width)),
                    float(np.float32(np.float32(pixels_i[k, 1]) / height)),
                ]
            )
            pos.append([float(pts3d_i[k, 0]), float(pts3d_i[k, 1]), float(pts3d_i[k, 2])])
        j["marker"] = OrderedDict((("pix", pix), ("pos", pos)))

        j_cameras[rootname] = j

    with open(filepath, "w", encoding="utf-8") as fout:
        json.dump(OrderedDict((("cameras", j_cameras),)), fout, indent=4)
        fout.write("\n")

    return True


def read_from_json(
    filepath: str,
) -> Tuple[bool, List[Camera], List[str], List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """``ReadFromJson``: ``(ok, cameras, names, pixels, pts3d, sizes)``."""
    cameras: List[Camera] = []
    names: List[str] = []
    pixels: List[np.ndarray] = []
    pts3d: List[np.ndarray] = []
    sizes: List[Tuple[int, int]] = []

    j = _read_json_file(filepath)
    if not j:
        PLOGE(f"JSON File not exists or cannot be opened: {filepath}")
        return False, cameras, names, pixels, pts3d, sizes

    try:
        j_cameras = j["cameras"]
        # nlohmann::json orders object keys ascending
        for name in sorted(j_cameras.keys()):
            value = j_cameras[name]
            cam = Camera()
            cam.K = _memcpy_into(cam.K, value["K"])
            cam.R = _memcpy_into(cam.R, value["R"])
            cam.t = _memcpy_into(cam.t, value["t"])
            cam.dist = _memcpy_into(cam.dist, value["dist"])

            width = int(value["res"][0])
            height = int(value["res"][1])

            pixels_std = np.asarray(value["marker"]["pix"], dtype=np.float64).reshape(-1, 2)
            pts3d_std = np.asarray(value["marker"]["pos"], dtype=np.float64).reshape(-1, 3)

            # convert pix to normal scale (cv::Point2f -> float precision)
            pixs = np.empty_like(pixels_std)
            pixs[:, 0] = (width * pixels_std[:, 0]).astype(np.float32)
            pixs[:, 1] = (height * pixels_std[:, 1]).astype(np.float32)

            names.append(name)
            pixels.append(pixs)
            pts3d.append(pts3d_std)
            cameras.append(cam)
            sizes.append((width, height))
        return True, cameras, names, pixels, pts3d, sizes
    except (KeyError, ValueError, TypeError) as exc:
        PLOGE(f"Exception caught: {exc}")
        return False, [], [], [], [], []


def read_cam_from_json(filepath: str, names: Sequence[str]) -> Tuple[bool, List[Camera]]:
    """``ReadCamFromJson``: cameras aligned with ``names`` (by rootname)."""
    cameras = [Camera() for _ in names]

    j = _read_json_file(filepath)
    if not j:
        PLOGE(f"JSON File not exists or cannot be opened: {filepath}")
        return False, cameras

    try:
        j_cameras = j["cameras"]
        for i, name in enumerate(names):
            rootname, _ = splitext(name)
            if rootname in j_cameras:
                value = j_cameras[rootname]
                cameras[i].K = _memcpy_into(cameras[i].K, value["K"])
                cameras[i].R = _memcpy_into(cameras[i].R, value["R"])
                cameras[i].t = _memcpy_into(cameras[i].t, value["t"])
                cameras[i].dist = _memcpy_into(cameras[i].dist, value["dist"])
            else:
                PLOGE(f"Cannot find {rootname} in file: {filepath}")
                raise RuntimeError("Cannot find camera parameters in json file")
        return True, cameras
    except (KeyError, ValueError, TypeError, RuntimeError) as exc:
        PLOGE(f"Exception caught: {exc}")
        return False, cameras


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------
def load_imgs_and_features(
    img_dir: str, feature_dir: str
) -> Tuple[bool, List[str], List[ImageFeatures], List[Tuple[int, int]]]:
    """``LoadImgsAndFeatures``: ``(ok, fnames, features, sizes)``."""
    fpaths = sorted(listdir(img_dir))

    fnames: List[str] = []
    features: List[ImageFeatures] = []
    sizes: List[Tuple[int, int]] = []

    for fpath in fpaths:
        fname = basename(fpath)
        _, ext = splitext(fname)

        if ext not in VALID_IMG_EXTS:
            continue
        if fname == "mask.png":
            continue

        image = cv2.imread(fpath)
        if image is None or image.size == 0:
            continue

        size = (image.shape[1], image.shape[0])  # cv::Size(width, height)
        feature = ImageFeatures(img_idx=len(fnames), img_size=size)
        feature_path = feature_dir + "/" + fname + ".txt"
        feature.keypoints, feature.descriptors = read_colmap_features(feature_path)

        PLOGI(f"Index: {len(fnames)}, image: {fname}")
        fnames.append(fname)
        features.append(feature)
        sizes.append(size)

    num_images = len(fnames)
    if num_images < 2:
        PLOGE(f"Images number not enough (< 2): {num_images}")
        return False, fnames, features, sizes

    return True, fnames, features, sizes


def cal_homography(kpts1: np.ndarray, kpts2: np.ndarray, matches: np.ndarray, ransac_thresh: float):
    """``CalHomography``: H maps points of image1 onto image2."""
    if matches.shape[0] < 4:
        return None
    ref_pts = kpts1[matches[:, 0]].astype(np.float32)
    src_pts = kpts2[matches[:, 1]].astype(np.float32)
    H, _mask = cv2.findHomography(ref_pts, src_pts, cv2.RANSAC, ransac_thresh)
    return H


def cal_matching_score(num_matches: int, max_num_matches: int) -> float:
    """``CalMatchingScore``"""
    assert num_matches >= 0 and max_num_matches >= 0
    if num_matches >= max_num_matches:
        return 1.0
    return float(np.float32(num_matches) / np.float32(max_num_matches))


def load_matches_info(
    matches_path: str, fnames: Sequence[str], features: Sequence[ImageFeatures]
) -> Tuple[bool, List[MatchesInfo]]:
    """``LoadMatchesInfo``: dense ``num_images * num_images`` table (row-major)."""
    assert len(fnames) == len(features)

    pairs_matches, img_pairs_name = read_colmap_matches(matches_path)

    num_images = len(fnames)
    matches_info: List[MatchesInfo] = [MatchesInfo() for _ in range(num_images * num_images)]

    for i in range(len(pairs_matches)):
        index_i = find_img_index(fnames, img_pairs_name[i][0])
        index_j = find_img_index(fnames, img_pairs_name[i][1])
        if index_i < 0 or index_j < 0:
            continue

        H_j_i = cal_homography(
            features[index_i].keypoints, features[index_j].keypoints, pairs_matches[i], RANSAC_THRESH
        )

        info = MatchesInfo()
        info.matches = pairs_matches[i]
        info.H = H_j_i
        info.inliers_mask = np.ones(pairs_matches[i].shape[0], dtype=np.uint8)
        info.num_inliers = int(pairs_matches[i].shape[0])
        info.confidence = cal_matching_score(int(pairs_matches[i].shape[0]), MAX_NUM_MATCHES)
        info.src_img_idx = index_i
        info.dst_img_idx = index_j
        matches_info[index_i * num_images + index_j] = info

    return True, matches_info


def load_annotation(
    annot_path: str, fnames: Sequence[str]
) -> Tuple[bool, List[np.ndarray], List[np.ndarray]]:
    """``LoadAnnotation``: ``(ok, pixels, pts3d)`` indexed like ``fnames``."""
    ok, _gt_cameras, gt_names, gt_pixels, gt_pts3d, _gt_sizes = read_from_json(annot_path)
    if not ok:
        return False, [], []

    num_images = len(fnames)
    pixels: List[np.ndarray] = [np.zeros((0, 2)) for _ in range(num_images)]
    pts3d: List[np.ndarray] = [np.zeros((0, 3)) for _ in range(num_images)]

    for i in range(len(gt_names)):
        idx = find_img_index(fnames, gt_names[i])
        if idx == -1:
            continue
        pixels[idx] = gt_pixels[i]
        pts3d[idx] = gt_pts3d[i]

    return True, pixels, pts3d


def save_registered_cam(
    cameras: Sequence[Camera],
    reg_image_ids: Set[int],
    fnames: Sequence[str],
    pixels: Sequence[np.ndarray],
    pts3d: Sequence[np.ndarray],
    out_path: str,
) -> None:
    """``SaveRegisteredCam``"""
    cameras_registered: List[Camera] = []
    names_registered: List[str] = []
    pixels_registered: List[np.ndarray] = []
    pts3d_registered: List[np.ndarray] = []

    for i in range(len(cameras)):
        if i not in reg_image_ids:
            PLOGI(f"Filter out failed image #{i}: {fnames[i]}")
            continue
        cameras_registered.append(cameras[i])
        names_registered.append(fnames[i])
        pixels_registered.append(pixels[i])
        pts3d_registered.append(pts3d[i])

    save_to_json(cameras_registered, names_registered, pixels_registered, pts3d_registered, out_path)


def find_img_index(fnames: Sequence[str], fname: str) -> int:
    """``FindImgIndex``: compare basenames without extension."""
    name_wo_ext, _ = splitext(fname)
    for i, name_i in enumerate(fnames):
        name_wo_ext_i, _ = splitext(name_i)
        if name_wo_ext_i == name_wo_ext:
            return i
    return -1
