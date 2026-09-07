"""Phase-0 sanity checks: camera model, JSON schema round-trip, track statistics."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ptzcalib import data_io  # noqa: E402
from ptzcalib.logging_utils import init_logging  # noqa: E402
from ptzcalib.os_path import basename, splitext  # noqa: E402
from ptzcalib.tracks import TracksBuilder, length  # noqa: E402
from ptzcalib.types import Camera, rodrigues_batch  # noqa: E402

DATA = os.environ.get("PTZ_DATA", "../PTZ-Calib/data")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = DATA if os.path.isabs(DATA) else os.path.normpath(os.path.join(HERE, "..", DATA))

_failures = []


def check(name, cond, detail=""):
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        _failures.append(name)


def test_os_path():
    check("splitext('a.jpg')", splitext("a.jpg") == ("a", ".jpg"))
    check("splitext('scene_01-1.jpg')", splitext("scene_01-1.jpg") == ("scene_01-1", ".jpg"))
    check("splitext('dir.d/name')", splitext("dir.d/name") == ("dir.d/name", ""))
    check("basename('/a/b/c.png')", basename("/a/b/c.png") == "c.png")


def test_camera_vector_roundtrip():
    rng = np.random.default_rng(0)
    cam = Camera()
    cam.K = np.array([[1234.5, 0, 640.0], [0, 1234.5, 360.0], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(rng.normal(size=(3, 1)) * 0.3)
    cam.R = R
    cam.t = rng.normal(size=(3, 1))
    cam.dist = np.array([[-0.12], [0.0], [0.0], [0.0], [0.0]])

    v = cam.to_vector()
    cam2 = Camera.from_vector_new(v)
    err = max(
        np.max(np.abs(cam.K - cam2.K)),
        np.max(np.abs(cam.R - cam2.R)),
        np.max(np.abs(cam.t - cam2.t)),
        np.max(np.abs(cam.dist - cam2.dist)),
    )
    check("Camera to_vector/from_vector round-trip", err < 1e-12, f"max err {err:.2e}")


def test_rodrigues_batch():
    rng = np.random.default_rng(1)
    rvecs = np.vstack([rng.normal(size=(16, 3)) * 1.5, np.zeros((1, 3))])
    Rb = rodrigues_batch(rvecs)
    err = 0.0
    for i, r in enumerate(rvecs):
        Rcv, _ = cv2.Rodrigues(r.reshape(3, 1))
        err = max(err, float(np.max(np.abs(Rcv - Rb[i]))))
    check("rodrigues_batch matches cv2.Rodrigues", err < 1e-12, f"max err {err:.2e}")


def test_projection_roundtrip():
    """pixel -> ray -> pixel for the pinhole (no-distortion) model used by PTZRay."""
    rng = np.random.default_rng(2)
    K = np.array([[900.0, 0, 600.0], [0, 900.0, 400.0], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(np.array([[0.1], [-0.2], [0.05]]))
    uv = np.column_stack([rng.uniform(0, 1200, 64), rng.uniform(0, 800, 64)])

    # ray = R^-1 K^-1 [u, v, 1], normalised (Pix2Ray)
    pts = np.column_stack([uv, np.ones(len(uv))])
    ray = pts @ np.linalg.inv(K).T @ np.linalg.inv(R).T
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)

    p = ray @ R.T
    back = np.column_stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]])
    err = float(np.max(np.abs(back - uv)))
    check("pixel -> ray -> pixel round-trip", err < 1e-9, f"max err {err:.2e}")


def test_json_roundtrip():
    gt = f"{DATA}/synthetic/gt/scene_01.json"
    if not os.path.exists(gt):
        check("JSON read/write round-trip", False, f"missing {gt}")
        return
    ok, cameras, names, pixels, pts3d, sizes = data_io.read_from_json(gt)
    check("read_from_json(gt/scene_01)", ok and len(cameras) == 180, f"{len(cameras)} cameras")

    tmp = os.path.join(HERE, "_roundtrip.json")
    data_io.save_to_json(cameras, names, pixels, pts3d, tmp)
    ok2, cameras2, names2, pixels2, pts3d2, sizes2 = data_io.read_from_json(tmp)
    os.remove(tmp)

    kerr = max(float(np.max(np.abs(a.K - b.K))) for a, b in zip(cameras, cameras2))
    rerr = max(float(np.max(np.abs(a.R - b.R))) for a, b in zip(cameras, cameras2))
    perr = max(float(np.max(np.abs(a - b))) for a, b in zip(pixels, pixels2))
    xerr = max(float(np.max(np.abs(a - b))) for a, b in zip(pts3d, pts3d2))
    # marker.pix is normalised by the resolution and stored through float32 in the
    # C++ writer, so a ~1e-4 px round-trip error is the expected (matching) behaviour
    check(
        "JSON write/read round-trip",
        ok2 and names == names2 and sizes == sizes2 and max(kerr, rerr, xerr) < 1e-9 and perr < 1e-3,
        f"K {kerr:.1e} R {rerr:.1e} pix {perr:.1e} pos {xerr:.1e}",
    )


def test_tracks_scene01():
    img_dir = f"{DATA}/synthetic/offline/scene_01"
    feat_dir = f"{DATA}/synthetic/offline_matches/scene_01"
    if not os.path.isdir(img_dir):
        check("scene_01 track statistics", False, f"missing {img_dir}")
        return
    cv2.setRNGSeed(0)
    ok, fnames, features, _sizes = data_io.load_imgs_and_features(img_dir, feat_dir)
    check("load_imgs_and_features(scene_01)", ok and len(fnames) == 30, f"{len(fnames)} images")

    ok, matches_info = data_io.load_matches_info(f"{feat_dir}/pairs_matches.txt", fnames, features)
    n_pairs = sum(1 for m in matches_info if m.matches.shape[0] > 0)
    check("load_matches_info(scene_01)", ok and n_pairs == 390, f"{n_pairs} pairs")

    builder = TracksBuilder()
    builder.build(matches_info)
    builder.filter(4)
    tracks = builder.export_to_stl()
    total, mx, mn = length(tracks)
    check(
        "scene_01 track statistics",
        len(tracks) == 2216 and total == 12897 and mn == 4 and mx == 15,
        f"{len(tracks)} tracks, total {total}, min {mn}, max {mx}",
    )


if __name__ == "__main__":
    init_logging(False)
    logging.getLogger("ptzcalib").setLevel(logging.WARNING)

    test_os_path()
    test_camera_vector_roundtrip()
    test_rodrigues_batch()
    test_projection_roundtrip()
    test_json_roundtrip()
    test_tracks_scene01()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: {_failures}")
        sys.exit(1)
    print("all checks passed")
