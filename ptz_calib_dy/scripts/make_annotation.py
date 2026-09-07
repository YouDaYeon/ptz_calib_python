#!/usr/bin/env python
"""PLAN.md section 6-1 - survey CSV -> annotation JSON in the PTZ-Calib schema.

The 2D-3D correspondences are written in exactly the schema the reference
implementation uses (``PTZ-Calib/data/*/offline/<scene>/<scene>.json`` and
``gt/<scene>.json``), so ``run_ptz_ba.py -a`` consumes them unchanged::

    {"cameras": {"<image stem>": {
        "name": ..., "pos": [], "res": [W, H],
        "marker": {"pix": [[u/W, v/H], ...], "pos": [[X, Y, Z], ...]},
        "K": [], "R": [], "t": [], "dist": [], "distType": ""}}}

``pix`` is normalised by the resolution - ``data_io.read_from_json`` multiplies
``W``/``H`` back in.  Writing raw pixels here fails silently.

Three files are produced.  ``cam_id`` is the images-directory basename, the same
rule ``run_ptz_ba.py`` uses to name its output, so ``gt/<cam_id>.json`` pairs with
``<out>/<cam_id>.json`` exactly as ``run_ptzba_synthetic.sh`` pairs them:

==============================  ===========================================
``offline/<off_id>.json``       ``split=fit`` - georeferencing input (``-a``)
``gt/<off_id>.json``            ``split=holdout`` on offline frames
``gt/<on_id>.json``             ``split=holdout`` on online frames
==============================  ===========================================

Two gt files rather than one: the public ``gt/scene_01.json`` covers all 180
frames of a scene in a single namespace, but our offline and online sets reuse
the same 22 filenames (``ptz_frame_001.jpg`` exists in both), so a union file
would collide on the camera key.  Splitting by ``cam_id`` is the same convention
the run scripts already use.

The fit/holdout split lives in the CSV, not in the JSON: the public schema has no
field for it and never needs one, since it ships full pose GT.  Splitting by
*file* keeps the JSON identical in schema while still guaranteeing that no
holdout point can reach the georeferencing input.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptzcalib.data_io import VALID_IMG_EXTS  # noqa: E402


def read_csv(path: str) -> List[dict]:
    """CSV reader that skips ``#`` comment lines."""
    with open(path, "r", encoding="utf-8") as fin:
        lines = [ln for ln in fin if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def image_res(images_dir: str, fname: str) -> Tuple[int, int]:
    import cv2

    img = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image: {os.path.join(images_dir, fname)}")
    return img.shape[1], img.shape[0]


def list_images(images_dir: str) -> set:
    return {
        f
        for f in os.listdir(images_dir)
        if f != "mask.png" and os.path.splitext(f)[1].lower() in VALID_IMG_EXTS
    }


def build(
    points: Dict[str, dict],
    obs: List[dict],
    res_of: Dict[str, Tuple[int, int]],
) -> collections.OrderedDict:
    """Group observations by frame into the PTZ-Calib camera dict."""
    by_frame: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in obs:
        by_frame[r["fname"]].append(r)

    cameras = collections.OrderedDict()
    for fname in sorted(by_frame):
        stem = os.path.splitext(fname)[0]
        width, height = res_of[fname]
        pix, pos = [], []
        for r in sorted(by_frame[fname], key=lambda x: x["point_id"]):
            p = points[r["point_id"]]
            pix.append([float(r["u"]) / width, float(r["v"]) / height])
            pos.append([float(p["X"]), float(p["Y"]), float(p["Z"])])
        cameras[stem] = collections.OrderedDict(
            [
                ("name", stem),
                ("pos", []),
                ("res", [width, height]),
                ("marker", collections.OrderedDict([("pix", pix), ("pos", pos)])),
                ("K", []),
                ("R", []),
                ("t", []),
                ("dist", []),
                ("distType", ""),
            ]
        )
    return cameras


def write_json(path: str, cameras: collections.OrderedDict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump({"cameras": cameras}, fout, indent=4)
        fout.write("\n")
    n_mark = sum(len(c["marker"]["pix"]) for c in cameras.values())
    print(f"  {path}: {len(cameras)} frames, {n_mark} markers")


def main() -> int:
    p = argparse.ArgumentParser(description="survey CSV -> PTZ-Calib annotation JSON")
    p.add_argument("--survey_dir", required=True, help="directory holding points.csv / observations.csv")
    p.add_argument("--offline_images", required=True)
    p.add_argument("--online_images", default="")
    p.add_argument("-o", "--output_root", required=True, help="dataset root, e.g. data/infiniq")
    args = p.parse_args()

    # same rule as run_ptz_ba.py: cam_id is the images-directory basename
    off_id = os.path.basename(os.path.normpath(args.offline_images))
    on_id = os.path.basename(os.path.normpath(args.online_images)) if args.online_images else ""

    points = {r["point_id"]: r for r in read_csv(os.path.join(args.survey_dir, "points.csv"))}
    obs = read_csv(os.path.join(args.survey_dir, "observations.csv"))

    # ---- validation: a silent mismatch here produces a plausible but wrong metric
    errs: List[str] = []
    for r in obs:
        if r["point_id"] not in points:
            errs.append(f"observation references unknown point_id {r['point_id']}")
    unobserved = set(points) - {r["point_id"] for r in obs}
    if unobserved:
        errs.append(f"points never observed: {sorted(unobserved)}")
    for pid, r in points.items():
        if r["split"] not in ("fit", "holdout"):
            errs.append(f"{pid}: split must be fit|holdout, got {r['split']!r}")

    imgs = {"offline": list_images(args.offline_images)}
    if args.online_images:
        imgs["online"] = list_images(args.online_images)
    for r in obs:
        if r["set"] not in imgs:
            errs.append(f"unknown set {r['set']!r} (expected {sorted(imgs)})")
        elif r["fname"] not in imgs[r["set"]]:
            errs.append(f"{r['set']}/{r['fname']} is not an image in that directory")
    if errs:
        for e in sorted(set(errs)):
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    dirs = {"offline": args.offline_images, "online": args.online_images}
    res_of = {
        r["fname"]: image_res(dirs[r["set"]], r["fname"]) for r in obs
    }
    for r in obs:
        w, h = res_of[r["fname"]]
        u, v = float(r["u"]), float(r["v"])
        if not (0.0 <= u < w and 0.0 <= v < h):
            print(f"ERROR: {r['fname']} {r['point_id']}: ({u}, {v}) outside {w}x{h}", file=sys.stderr)
            return 1

    split_of = {pid: r["split"] for pid, r in points.items()}
    off_obs = [r for r in obs if r["set"] == "offline"]
    on_obs = [r for r in obs if r["set"] == "online"]

    print("writing annotation JSON (PTZ-Calib schema):")
    write_json(
        os.path.join(args.output_root, "offline", f"{off_id}.json"),
        build(points, [r for r in off_obs if split_of[r["point_id"]] == "fit"], res_of),
    )
    write_json(
        os.path.join(args.output_root, "gt", f"{off_id}.json"),
        build(points, [r for r in off_obs if split_of[r["point_id"]] == "holdout"], res_of),
    )
    if on_obs:
        write_json(
            os.path.join(args.output_root, "gt", f"{on_id}.json"),
            build(points, [r for r in on_obs if split_of[r["point_id"]] == "holdout"], res_of),
        )

    n_fit = sum(v == "fit" for v in split_of.values())
    print(f"points: {len(points)} (fit {n_fit}, holdout {len(points) - n_fit}), observations: {len(obs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
