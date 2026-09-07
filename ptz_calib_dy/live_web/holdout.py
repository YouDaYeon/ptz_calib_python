"""Holdout 2D->3D error from a *user-supplied* pixel, in metres.

This is ``scripts/eval_own.py::holdout_2d3d``'s metre error, one click at a
time: shoot a ray through the clicked pixel, intersect the surveyed point's
horizontal plane, and compare the result with the surveyed 3D point.

The pixel has to come from outside the calibration.  Projecting the survey
point to get one and back-projecting it again is the *identity* -
``geometry.points_to_pixels`` and ``geometry.pixels_to_plane`` are exact
inverses and every holdout point sits on ``Z = 0``, the plane the rays are
intersected with - so it returns 0.000 m for any point that lands inside the
frame, whatever the calibration is.  A human clicking the marker is an
independent observation; that is what ``eval_own`` scores against
(``marker.pix_raw``) and what this module expects.

Which surveyed point a click refers to can be named explicitly or left to
``nearest``: holdout points are 0.445 m apart, so the closest one is
unambiguous as long as the error stays well under half that.  Clicks that land
further out are flagged rather than silently scored against a neighbour.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import geometry

# Holdout points are on a 0.445 m tile grid.  Past this the click is closer to
# being between two markers than on one, so the automatic choice is not trustworthy.
AMBIGUOUS_M = 0.30


def targets(survey: Sequence[dict], split: str = "holdout") -> List[dict]:
    """Surveyed points a click can be scored against."""
    return [
        {"id": s["id"], "xyz": list(s["xyz"])}
        for s in survey
        if not split or s.get("split") == split
    ]


def _nearest(pts: np.ndarray, xyz: np.ndarray) -> Tuple[int, float]:
    d = np.linalg.norm(pts - xyz[None, :], axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def score(
    cam,
    uv: np.ndarray,
    tgts: Sequence[dict],
    names: Optional[Sequence[Optional[str]]] = None,
) -> dict:
    """Score clicked pixels against the surveyed holdout points.

    ``names[i]`` pins measurement ``i`` to that point id; ``None`` picks the
    nearest.  Each ray is intersected with **its own point's** Z, which is what
    ``eval_own`` does - found in two passes because the point is only known
    after a first back-projection.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if uv.shape[0] == 0:    # cv2.undistortPoints returns None on an empty array
        return {"points": [], "summary": None}
    if not tgts:
        return {"points": [], "summary": None, "reason": "no holdout survey points loaded"}

    xyz_of = {t["id"]: np.asarray(t["xyz"], dtype=np.float64) for t in tgts}
    pts = np.stack([xyz_of[t["id"]] for t in tgts])
    ids = [t["id"] for t in tgts]

    # pass 1: a provisional 3D point, on the plane the holdout points share
    z0 = float(np.median(pts[:, 2]))
    prov, prov_ok = geometry.pixels_to_plane(cam, uv, z0)

    out: List[dict] = []
    errors: List[float] = []
    for i in range(uv.shape[0]):
        item: dict = {"pixel": [float(uv[i, 0]), float(uv[i, 1])], "valid": False}
        pinned = names[i] if names and i < len(names) else None
        if pinned is not None and pinned not in xyz_of:
            item["reason"] = f"unknown holdout point '{pinned}'"
            out.append(item)
            continue
        if not prov_ok[i]:
            item["reason"] = "the ray never meets the floor plane in front of the camera (above the horizon?)"
            out.append(item)
            continue

        if pinned is not None:
            gt, gt_id, dist, auto = xyz_of[pinned], pinned, None, False
        else:
            j, dist = _nearest(pts, prov[i])
            gt, gt_id, auto = pts[j], ids[j], True

        # pass 2: re-intersect at the chosen point's own height
        xyz, ok = geometry.pixels_to_plane(cam, uv[i : i + 1], float(gt[2]))
        if not ok[0]:
            item["reason"] = "the ray never meets Z = %.3f in front of the camera" % gt[2]
            out.append(item)
            continue
        xyz = xyz[0]
        uv_gt, _in_front = geometry.points_to_pixels(cam, gt[None, :])

        item.update(
            {
                "valid": True,
                "target": gt_id,
                "target_auto": auto,
                "world_gt": gt.tolist(),
                "world": xyz.tolist(),
                "pixel_gt": [float(uv_gt[0, 0]), float(uv_gt[0, 1])],
                "error_m": float(np.linalg.norm(xyz - gt)),
                "dx_m": float(xyz[0] - gt[0]),
                "dy_m": float(xyz[1] - gt[1]),
                "dz_m": float(xyz[2] - gt[2]),
                "error_px": float(np.linalg.norm(uv_gt[0] - uv[i])),
            }
        )
        if auto:
            item["nearest_m"] = dist
            item["ambiguous"] = bool(dist > AMBIGUOUS_M)
        errors.append(item["error_m"])
        out.append(item)

    return {"points": out, "summary": summarise(np.asarray(errors)) if errors else None}


def summarise(err: np.ndarray) -> dict:
    return {
        "n": int(err.size),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "max": float(np.max(err)),
    }
