#!/usr/bin/env python
"""Numeric comparison of two PTZ-Calib result JSONs (python port vs. C++ baseline)."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    rel = Ra @ Rb.T
    cos = (np.trace(rel) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def compare(pred_path: str, ref_path: str) -> Dict[str, float]:
    jp, jr = read_json(pred_path), read_json(ref_path)
    keys_p = set(jp["cameras"])
    keys_r = set(jr["cameras"])
    common = sorted(keys_p & keys_r)

    df: List[float] = []
    dr: List[float] = []
    dt: List[float] = []
    dd: List[float] = []
    for k in common:
        a, b = jp["cameras"][k], jr["cameras"][k]
        df.append(abs(a["K"][0] - b["K"][0]))
        dr.append(rot_angle_deg(np.array(a["R"]).reshape(3, 3), np.array(b["R"]).reshape(3, 3)))
        dt.append(float(np.linalg.norm(np.array(a["t"]) - np.array(b["t"]))))
        dd.append(float(np.max(np.abs(np.array(a["dist"], dtype=float) - np.array(b["dist"], dtype=float)))))

    return {
        "n_common": len(common),
        "n_pred_only": len(keys_p - keys_r),
        "n_ref_only": len(keys_r - keys_p),
        "df_mean": float(np.mean(df)) if df else float("nan"),
        "df_max": float(np.max(df)) if df else float("nan"),
        "dR_mean": float(np.mean(dr)) if dr else float("nan"),
        "dR_max": float(np.max(dr)) if dr else float("nan"),
        "dt_mean": float(np.mean(dt)) if dt else float("nan"),
        "dt_max": float(np.max(dt)) if dt else float("nan"),
        "ddist_max": float(np.max(dd)) if dd else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare python-port output against the C++ baseline")
    ap.add_argument("--pred_dir", required=True, help="Python output directory")
    ap.add_argument("--ref_dir", required=True, help="C++ output directory")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.pred_dir) if f.endswith(".json"))
    hdr = (
        f"{'scene':<12}{'N':>4}{'+p':>4}{'+r':>4} | {'df mean':>9}{'df max':>9} | "
        f"{'dR mean':>9}{'dR max':>9} | {'dt mean':>9}{'dt max':>9} | {'ddist':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    agg: List[Dict[str, float]] = []
    for name in names:
        ref = os.path.join(args.ref_dir, name)
        if not os.path.exists(ref):
            continue
        r = compare(os.path.join(args.pred_dir, name), ref)
        agg.append(r)
        print(
            f"{name[:-5]:<12}{r['n_common']:>4}{r['n_pred_only']:>4}{r['n_ref_only']:>4} | "
            f"{r['df_mean']:>9.4f}{r['df_max']:>9.4f} | {r['dR_mean']:>9.5f}{r['dR_max']:>9.5f} | "
            f"{r['dt_mean']:>9.5f}{r['dt_max']:>9.5f} | {r['ddist_max']:>9.2e}"
        )
    if agg:
        print("-" * len(hdr))
        print(
            f"{'ALL':<12}{sum(a['n_common'] for a in agg):>4}"
            f"{sum(a['n_pred_only'] for a in agg):>4}{sum(a['n_ref_only'] for a in agg):>4} | "
            f"{np.mean([a['df_mean'] for a in agg]):>9.4f}{np.max([a['df_max'] for a in agg]):>9.4f} | "
            f"{np.mean([a['dR_mean'] for a in agg]):>9.5f}{np.max([a['dR_max'] for a in agg]):>9.5f} | "
            f"{np.mean([a['dt_mean'] for a in agg]):>9.5f}{np.max([a['dt_max'] for a in agg]):>9.5f} | "
            f"{np.max([a['ddist_max'] for a in agg]):>9.2e}"
        )


if __name__ == "__main__":
    main()
