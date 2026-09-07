#!/usr/bin/env python
"""PLAN.md section 5 - qualitative check: stitch the registered frames into a panorama.

If the estimated rotations and focal lengths are right, frames land on top of each
other and the seams line up; if they are wrong, the overlap ghosts or tears.  The
scene here is an indoor corridor with no surveyed ground plane, so a BEV render
(the `render_bev.py` slot in section 1-1) would show nothing useful - an
equirectangular panorama is the check that actually applies to pure rotation.

Blending is a plain mean of the contributing frames on purpose: averaging turns
misalignment into visible ghosting, whereas feathering would hide it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_cameras(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as fin:
        j = json.load(fin)
    out = {}
    for name, v in j["cameras"].items():
        out[name] = {
            "K": np.asarray(v["K"], dtype=np.float64).reshape(3, 3),
            "R": np.asarray(v["R"], dtype=np.float64).reshape(3, 3),
            "res": v["res"],
        }
    return out


def canvas_frame(cams: Dict[str, dict]) -> np.ndarray:
    """World -> canvas rotation that puts the panorama upright.

    PTZ-IBA fixes the world frame to whichever camera it registered first, so the
    raw world axes are arbitrary and the canvas can come out upside down.  Derive
    the canvas axes from the cameras instead: mean viewing direction becomes
    +Z, mean camera up becomes canvas up.  Rows are (right, down, forward).
    """
    fwd = np.zeros(3)
    up = np.zeros(3)
    for cam in cams.values():
        R = cam["R"]
        fwd += R.T @ np.array([0.0, 0.0, 1.0])
        up += R.T @ np.array([0.0, -1.0, 0.0])

    e_z = fwd / np.linalg.norm(fwd)
    up -= (up @ e_z) * e_z
    e_y = -up / np.linalg.norm(up)  # canvas Y points down, like image coordinates
    e_x = np.cross(e_y, e_z)
    e_x /= np.linalg.norm(e_x)
    return np.stack([e_x, e_y, e_z], axis=0)


def angular_extent(cams: Dict[str, dict]) -> Tuple[float, float, float, float]:
    """Longitude/latitude bounds covered by the camera frusta (radians)."""
    lons: List[float] = []
    lats: List[float] = []
    for cam in cams.values():
        K, R = cam["K"], cam["R"]
        w, h = cam["res"]
        Kinv = np.linalg.inv(K)
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1], [w / 2, h / 2, 1]], dtype=np.float64)
        for c in corners:
            d = R.T @ (Kinv @ c)
            d /= np.linalg.norm(d)
            lons.append(math.atan2(d[0], d[2]))
            lats.append(math.asin(max(-1.0, min(1.0, -d[1]))))

    lon = np.asarray(lons)
    # unwrap around the mean direction so a set straddling +-pi still gives a tight range
    mean = math.atan2(float(np.sin(lon).mean()), float(np.cos(lon).mean()))
    lon = mean + np.angle(np.exp(1j * (lon - mean)))
    return float(lon.min()), float(lon.max()), float(min(lats)), float(max(lats))


def render(cams: Dict[str, dict], img_dir: str, width: int, margin_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    lon0, lon1, lat0, lat1 = angular_extent(cams)
    m = math.radians(margin_deg)
    lon0, lon1, lat0, lat1 = lon0 - m, lon1 + m, lat0 - m, lat1 + m

    height = max(1, int(round(width * (lat1 - lat0) / (lon1 - lon0))))
    lon = np.linspace(lon0, lon1, width)
    lat = np.linspace(lat1, lat0, height)  # row 0 is the top of the canvas = the largest latitude
    LON, LAT = np.meshgrid(lon, lat)

    # world ray for every output pixel
    cl = np.cos(LAT)
    dirs = np.stack([cl * np.sin(LON), -np.sin(LAT), cl * np.cos(LON)], axis=-1)  # (H, W, 3)

    acc = np.zeros((height, width, 3), dtype=np.float64)
    cnt = np.zeros((height, width), dtype=np.float64)
    outline = np.zeros((height, width, 3), dtype=np.uint8)

    for name in sorted(cams):
        path = None
        for ext in (".jpg", ".png", ".jpeg", ".bmp", ".tiff"):
            cand = os.path.join(img_dir, name + ext)
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            print(f"  [skip] no image for {name}")
            continue
        img = cv2.imread(path)
        if img is None:
            continue

        K, R = cams[name]["K"], cams[name]["R"]
        h, w = img.shape[:2]

        d_cam = dirs @ R.T  # (H, W, 3)
        z = d_cam[..., 2]
        front = z > 1e-9
        zz = np.where(front, z, 1.0)
        u = (K[0, 0] * d_cam[..., 0] / zz) + K[0, 2]
        v = (K[1, 1] * d_cam[..., 1] / zz) + K[1, 2]

        inside = front & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if not inside.any():
            continue

        sampled = cv2.remap(
            img,
            u.astype(np.float32),
            v.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        acc[inside] += sampled[inside]
        cnt[inside] += 1.0

        edge = (inside.astype(np.uint8) * 255) - cv2.erode(inside.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        outline[edge > 0] = (0, 255, 255)
        print(f"  [warp] {name}  covers {100.0 * inside.mean():.1f}% of the canvas")

    valid = cnt > 0
    pano = np.zeros((height, width, 3), dtype=np.uint8)
    pano[valid] = (acc[valid] / cnt[valid, None]).clip(0, 255).astype(np.uint8)

    overlap = np.zeros((height, width, 3), dtype=np.uint8)
    overlap[..., 1] = np.clip(cnt * 40, 0, 255).astype(np.uint8)  # brighter = more frames agree here
    seams = pano.copy()
    seams[outline.any(axis=2)] = (0, 255, 255)

    print(f"  canvas {width}x{height}, covered {100.0 * valid.mean():.1f}%, max overlap {int(cnt.max())} frames")
    return pano, seams


def main() -> int:
    p = argparse.ArgumentParser(description="Stitch registered frames into an equirectangular panorama")
    p.add_argument("-i", "--images", required=True)
    p.add_argument("--params", required=True, help="run_ptz_ba.py output JSON")
    p.add_argument("-o", "--output", required=True, help="output image path (.jpg/.png)")
    p.add_argument("--width", type=int, default=4000)
    p.add_argument("--margin_deg", type=float, default=2.0)
    args = p.parse_args()

    cams = load_cameras(args.params)
    print(f"stitching {len(cams)} registered cameras from {args.images}")

    # re-express every rotation in the canvas frame: d_cam = R d_world = (R W^T) d_canvas
    W = canvas_frame(cams)
    for cam in cams.values():
        cam["R"] = cam["R"] @ W.T
    pano, seams = render(cams, args.images, args.width, args.margin_deg)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    cv2.imwrite(args.output, pano)
    root, ext = os.path.splitext(args.output)
    cv2.imwrite(root + "_seams" + ext, seams)
    print(f"wrote {args.output} and {root}_seams{ext}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
