"""2D <-> 3D for a calibrated frame.

A single view cannot recover depth, so 2D->3D is only defined against a known
surface.  The scene's survey points all lie on the floor (``Z = 0``, metres,
+X East / +Y North / +Z Up), so a clicked pixel is back-projected to a ray and
intersected with the horizontal plane ``Z = plane_z``.  This is the same
computation ``scripts/eval_own.py::_click_ray_on_plane`` is scored with, so
what the browser shows is exactly what ``eval_report.json`` measures.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ptzcalib.types import Camera


def camera_center(cam: Camera) -> np.ndarray:
    """``C = -R^T t`` - camera position in world coordinates."""
    return (-cam.R.T @ cam.t).ravel()


def pixels_to_rays(cam: Camera, uv: np.ndarray) -> np.ndarray:
    """Undistort pixels and turn them into unnormalised world-frame directions."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(uv, cam.K, cam.dist.reshape(-1)).reshape(-1, 2)
    ray_cam = np.concatenate([undist, np.ones((undist.shape[0], 1))], axis=1)
    return (cam.R.T @ ray_cam.T).T


def pixels_to_plane(
    cam: Camera, uv: np.ndarray, plane_z: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect the click rays with ``Z = plane_z``.

    Returns ``(xyz (N,3), valid (N,))``.  A point is invalid when the ray is
    parallel to the plane, hits it behind the camera, or lands behind the image
    plane - i.e. when the click is above the horizon.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    n = uv.shape[0]
    ray_w = pixels_to_rays(cam, uv)
    C = camera_center(cam)

    dz = ray_w[:, 2]
    valid = np.abs(dz) > 1e-12
    lam = np.zeros(n, dtype=np.float64)
    lam[valid] = (plane_z - C[2]) / dz[valid]
    xyz = C[None, :] + lam[:, None] * ray_w

    x_cam = (cam.R @ xyz.T + cam.t).T
    valid = valid & (lam > 0) & (x_cam[:, 2] > 0.0)
    return xyz, valid


def points_to_pixels(cam: Camera, pts3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward projection (with distortion) - the inverse check of the above."""
    pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
    if pts3d.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=bool)
    proj, _ = cv2.projectPoints(pts3d, cam.rvec(), cam.t, cam.K, cam.dist.reshape(-1))
    uv = proj.reshape(-1, 2)
    in_front = (cam.R @ pts3d.T + cam.t)[2, :] > 0.0
    return uv, in_front


def horizon_v(cam: Camera, width: int, num: int = 64) -> Optional[np.ndarray]:
    """Image rows where the floor plane escapes to infinity, per column.

    Used by the UI to grey out the region where a click has no floor
    intersection instead of silently returning a nonsense coordinate.
    """
    us = np.linspace(0.0, float(width - 1), num)
    out = np.full(num, np.nan)
    # binary search per column on "does the ray still meet the plane"
    for i, u in enumerate(us):
        lo, hi = 0.0, 4000.0
        uv = np.stack([np.full(2, u), np.array([lo, hi])], axis=1)
        _, ok = pixels_to_plane(cam, uv)
        if ok[1] and not ok[0]:
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                _, okm = pixels_to_plane(cam, np.array([[u, mid]]))
                if okm[0]:
                    hi = mid
                else:
                    lo = mid
            out[i] = hi
    if np.all(np.isnan(out)):
        return None
    return np.stack([us, out], axis=1)
