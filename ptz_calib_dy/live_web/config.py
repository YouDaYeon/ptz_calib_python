"""Paths and tunables for the live verification app.

Defaults point at the ``infiniq`` bundle that ``EVALUATION.md`` scores:
the georeferenced offline solve with distortion
(``output/output-infiniq-offline-georef-dist/offline.json``) plus the
SuperPoint features written next to the reference images.  Every value can be
overridden on the command line (see :mod:`live_web.app`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent  # ptz_calib_dy/


@dataclass
class Config:
    # --- reference bundle (offline stage output) --------------------------
    ref_images: str = str(ROOT / "data/infiniq/offline_registered")
    ref_features: str = str(ROOT / "data/infiniq/offline_matches")
    ref_params: str = str(ROOT / "output/output-infiniq-offline-georef-dist/offline.json")
    mask: str = str(ROOT / "data/infiniq/offline/mask.png")
    survey_points: str = str(ROOT / "data/infiniq/survey/points.csv")

    # --- live source ------------------------------------------------------
    rtsp_url: str = ""
    env_file: str = str(ROOT / ".env")

    # --- online stage (must mirror the offline run) -----------------------
    dist: bool = True          # FactorType.FDist - reference was solved with --dist
    max_iter: int = 200        # run_ptz_reloc.MAX_ITER
    max_reproj_error: float = 100.0  # run_ptz_reloc.MAX_REPROJ_ERROR

    # --- matching (must mirror data/infiniq/*_matches/match_report.json) ---
    device: str = ""           # "" -> cuda if available
    max_keypoints: int = 2048
    keypoint_threshold: float = 0.01
    nms_radius: int = 4
    resize: int = -1           # -1 keeps the original resolution
    conf_thresh: float = 0.5
    min_inliers: int = 40
    superglue_weights: str = "outdoor"

    # --- 2D->3D -----------------------------------------------------------
    plane_z: float = 0.0       # floor, world Z=0 (survey/points.csv)

    # --- PTZ control (ONVIF) ----------------------------------------------
    # Host and credentials come from the same .env as the RTSP URL; the port is
    # ONVIF_PORT (80 on this camera - the SDK port 3000 refuses connections).
    ptz_enabled: bool = True
    ptz_profile: str = ""      # "" -> first media profile that has a PTZConfiguration
    ptz_speed: float = 0.5     # 0..1 in the ONVIF generic speed space

    # --- view guard -------------------------------------------------------
    # A calibration stays valid as long as the camera has not moved, so clicks
    # are allowed straight on the live view.  Phase correlation against the
    # calibrated frame says whether it still points the same way.  Measured on
    # this camera: a static scene with people walking through drifts <= 0.71 px,
    # a real 3 px pan reads 2.6 px, a 2 % zoom reads 1.8 px -> 3 px separates them.
    drift_tolerance_px: float = 3.0
    guard_width: int = 480     # signature width; phase correlation costs ~3 ms

    # --- server -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8888
    stream_width: int = 1280   # MJPEG downscale; clicks stay in original pixels
    stream_quality: int = 70
    stream_fps: float = 15.0
    seed: int = 0

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        env = os.getenv("LIVE_DEVICE", "").strip()
        if env:
            return env
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
