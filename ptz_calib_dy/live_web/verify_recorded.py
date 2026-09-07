#!/usr/bin/env python
"""Prove the live path computes the same thing the offline evaluation scores.

The web app re-does matching itself (a live frame has no ``pairs_matches.txt``),
so it could quietly drift from ``scripts/run_ptz_reloc.py``.  This replays the
recorded test set through :class:`live_web.reloc.RelocEngine` - image in,
camera out - and diffs the result against the saved
``output-infiniq-online-georef-dist/online.json``, whose accuracy
``eval_own.py`` already measured (holdout 2D-3D: ~12 px / ~5 cm median).

    .venv-match/bin/python -m live_web.verify_recorded \\
        --test_images data/infiniq/online \\
        --recorded output/output-infiniq-online-georef-dist/online.json

Frames the recorded run left unregistered are reported separately: the live
path searches all 20 registered references, while the recorded run could only
use the top-3 written by ``extract_match.py --mode online`` - so the live path
legitimately registers a few frames the recorded one dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_web import geometry  # noqa: E402
from live_web.config import Config, ROOT  # noqa: E402
from live_web.reloc import RelocEngine  # noqa: E402


def angle_between(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Degrees between two rotations."""
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main() -> int:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--test_images", default=str(ROOT / "data/infiniq/online"))
    p.add_argument("--recorded", default=str(ROOT / "output/output-infiniq-online-georef-dist/online.json"))
    p.add_argument("--ref_images", default=d.ref_images)
    p.add_argument("--ref_features", default=d.ref_features)
    p.add_argument("--ref_params", default=d.ref_params)
    p.add_argument("--mask", default=d.mask)
    p.add_argument("--device", default="")
    p.add_argument("--limit", type=int, default=0, help="only the first N frames")
    p.add_argument("-o", "--output", default="", help="write the per-frame diff as JSON")
    args = p.parse_args()

    cfg = Config(
        ref_images=args.ref_images,
        ref_features=args.ref_features,
        ref_params=args.ref_params,
        mask=args.mask,
        device=args.device,
    )
    engine = RelocEngine(cfg)

    with open(args.recorded, "r", encoding="utf-8") as fin:
        recorded = json.load(fin)["cameras"]

    fnames = engine.em.list_images(args.test_images)
    if args.limit:
        fnames = fnames[: args.limit]

    rows = []
    n_both, n_live_only, n_rec_only, n_neither = 0, 0, 0, 0
    df, dang, dpos = [], [], []

    for fname in fnames:
        frame = cv2.imread(os.path.join(args.test_images, fname))
        if frame is None:
            continue
        res = engine.calibrate(frame)

        stem = os.path.splitext(fname)[0]
        rec = recorded.get(stem)
        # run_ptz_reloc writes an identity camera for frames it failed on
        rec_ok = bool(rec) and float(np.asarray(rec["K"]).reshape(3, 3)[0, 0]) > 1.5

        row = {"frame": fname, "live_ok": bool(res["ok"]), "recorded_ok": rec_ok}
        if res["ok"]:
            row["best_ref"] = res["best_ref"]
            row["num_matches"] = res["num_matches"]
            row["reproj_px"] = round(res["reproj_error_px"], 3)

        if res["ok"] and rec_ok:
            n_both += 1
            cam = res["_camera"]
            Kr = np.asarray(rec["K"], dtype=np.float64).reshape(3, 3)
            Rr = np.asarray(rec["R"], dtype=np.float64).reshape(3, 3)
            tr = np.asarray(rec["t"], dtype=np.float64).reshape(3, 1)
            row["d_focal_px"] = float(abs(cam.K[0, 0] - Kr[0, 0]))
            row["d_rot_deg"] = angle_between(cam.R, Rr)
            row["d_pos_m"] = float(
                np.linalg.norm(geometry.camera_center(cam) - (-Rr.T @ tr).ravel())
            )
            df.append(row["d_focal_px"])
            dang.append(row["d_rot_deg"])
            dpos.append(row["d_pos_m"])
        elif res["ok"]:
            n_live_only += 1
        elif rec_ok:
            n_rec_only += 1
            row["reason"] = res.get("reason", "")
        else:
            n_neither += 1

        rows.append(row)
        flag = "==" if (res["ok"] and rec_ok) else ("L " if res["ok"] else ("R " if rec_ok else "-- "))
        extra = ""
        if res["ok"] and rec_ok:
            extra = f"  df={row['d_focal_px']:.2e}px  dR={row['d_rot_deg']:.2e}deg  dC={row['d_pos_m']:.2e}m"
        print(f"[{flag}] {fname}{extra}")

    def summary(vals, label, unit):
        if not vals:
            return f"{label}: n/a"
        a = np.asarray(vals)
        return f"{label}: max {a.max():.3e} {unit}  median {np.median(a):.3e} {unit}"

    print("-" * 70)
    print(f"frames                       : {len(rows)}")
    print(f"both registered              : {n_both}")
    print(f"live only (recorded dropped) : {n_live_only}")
    print(f"recorded only (live dropped) : {n_rec_only}")
    print(f"neither                      : {n_neither}")
    print(summary(df, "focal difference   ", "px"))
    print(summary(dang, "rotation difference", "deg"))
    print(summary(dpos, "position difference", "m"))
    print("-" * 70)

    report = {
        "test_images": os.path.abspath(args.test_images),
        "recorded": os.path.abspath(args.recorded),
        "ref_params": os.path.abspath(args.ref_params),
        "counts": {
            "frames": len(rows),
            "both": n_both,
            "live_only": n_live_only,
            "recorded_only": n_rec_only,
            "neither": n_neither,
        },
        "max_focal_diff_px": float(np.max(df)) if df else None,
        "max_rotation_diff_deg": float(np.max(dang)) if dang else None,
        "max_position_diff_m": float(np.max(dpos)) if dpos else None,
        "per_frame": rows,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fout:
            json.dump(report, fout, indent=2)
            fout.write("\n")
        print(f"report: {args.output}")

    return 0 if n_rec_only == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
