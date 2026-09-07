#!/usr/bin/env python
"""Live 2D->3D verification server.

    .venv-match/bin/python -m live_web.app            # from ptz_calib_dy/

Browse to http://<host>:8000.  "Capture & calibrate" freezes the current RTSP
frame, runs the online stage (PTZ-Reloc with distortion) against the
georeferenced reference bundle, and every click on that frame is reported as
(u, v) px and (X, Y, Z) m on the floor plane.

Endpoints
---------
``GET  /``                  UI
``GET  /api/status``        stream + engine + calibration state
``GET  /stream.mjpg``       live MJPEG
``GET  /api/snapshot.jpg``  the frozen, calibrated frame
``POST /api/calibrate``     run the online stage on the newest frame
``POST /api/unproject``     pixels -> world points on Z = plane_z
``POST /api/holdout``       clicked pixels -> 2D->3D error (m) vs the survey GT
``GET  /api/overlay``       survey points projected into the calibrated frame
``GET  /api/ptz``           ONVIF pose + preset list
``POST /api/ptz/move``      absolute pan/tilt/zoom move
``POST /api/ptz/preset``    recall a preset
``POST /api/ptz/stop``      stop whatever the camera is doing
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from live_web import geometry, holdout, ptz  # noqa: E402
from live_web.camera_stream import CameraStream, load_rtsp_url  # noqa: E402
from live_web.config import Config  # noqa: E402
from live_web.reloc import RelocEngine  # noqa: E402

app = Flask(__name__, static_folder=os.path.join(_HERE, "static"), static_url_path="/static")

CFG = Config()
STREAM: Optional[CameraStream] = None
ENGINE: Optional[RelocEngine] = None
PTZ: Optional[ptz.OnvifPtz] = None

_CALIB_LOCK = threading.Lock()
CALIB: Optional[dict] = None      # last *successful* calibration
LAST_ATTEMPT: Optional[dict] = None  # last attempt, successful or not


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _jpeg(frame: np.ndarray, width: int, quality: int) -> bytes:
    if 0 < width < frame.shape[1]:
        scale = width / float(frame.shape[1])
        frame = cv2.resize(
            frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _public_calib(c: dict) -> dict:
    """Strip the numpy/Camera payload before serialising."""
    return {k: v for k, v in c.items() if not k.startswith("_")}


# --- view guard -----------------------------------------------------------
_GUARD_CACHE: dict = {}


def _guard_signature(frame: np.ndarray) -> tuple:
    """Downscaled float32 grey + the scale back to original pixels."""
    scale = CFG.guard_width / float(frame.shape[1])
    small = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        (CFG.guard_width, max(1, int(round(frame.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    return small, 1.0 / scale


def _view_drift() -> Optional[dict]:
    """How far the live view has moved since the calibrated frame, in px.

    Phase correlation ignores local motion (people walking) and reports the
    global shift, which is exactly what a pan/tilt/zoom produces.  Cached per
    frame sequence number so repeated polls cost nothing.
    """
    with _CALIB_LOCK:
        calib = CALIB
    if calib is None or STREAM is None:
        return None
    ref = calib.get("_guard")
    if ref is None:
        return None

    frame, _recv_at, seq = STREAM.latest()
    if frame is None:
        return None

    key = (id(calib), seq)
    if _GUARD_CACHE.get("key") == key:
        return _GUARD_CACHE["value"]

    ref_small, scale = ref
    cur_small, _ = _guard_signature(frame)
    if cur_small.shape != ref_small.shape:
        out = {
            "shift_px": float("inf"),
            "response": 0.0,
            "stale": True,
            "reason": "live frame resolution changed",
        }
    else:
        window = cv2.createHanningWindow(ref_small.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(ref_small, cur_small, window)
        shift = float(np.hypot(dx, dy) * scale)
        out = {
            "shift_px": shift,
            "response": float(response),
            "stale": shift > CFG.drift_tolerance_px,
        }
    out["tolerance_px"] = CFG.drift_tolerance_px
    out["frame_seq"] = seq
    _GUARD_CACHE["key"] = key
    _GUARD_CACHE["value"] = out
    return out


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def api_status():
    frame, recv_at, seq = STREAM.latest() if STREAM else (None, None, 0)
    with _CALIB_LOCK:
        calib = _public_calib(CALIB) if CALIB else None
        attempt = _public_calib(LAST_ATTEMPT) if LAST_ATTEMPT else None

    return jsonify(
        {
            "stream": {
                "url": STREAM.url.split("@")[-1] if STREAM else "",
                "connected": bool(STREAM and STREAM.connected),
                "fps": round(STREAM.fps, 1) if STREAM else 0.0,
                "last_error": STREAM.last_error if STREAM else "not started",
                "recv_at": recv_at.isoformat(timespec="milliseconds") if recv_at else None,
                "seq": seq,
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])] if frame is not None else None,
            },
            "engine": {
                "ready": ENGINE is not None,
                "device": ENGINE.device if ENGINE else "",
                "num_refs": len(ENGINE.refs) if ENGINE else 0,
                "num_survey": len(ENGINE.survey) if ENGINE else 0,
                "offline_quality": ENGINE.offline_quality if ENGINE else None,
                "factor": "FDist" if CFG.dist else "F",
                "ref_params": CFG.ref_params,
                "plane_z": CFG.plane_z,
            },
            "ptz": {
                "enabled": CFG.ptz_enabled,
                "ready": bool(PTZ and PTZ.ready),
                "error": PTZ.last_error if PTZ else "",
            },
            "calibration": calib,
            "last_attempt": attempt,
            "view_drift": _view_drift(),
        }
    )


@app.route("/stream.mjpg")
def stream_mjpg():
    def generate():
        last_seq = -1
        period = 1.0 / max(CFG.stream_fps, 1.0)
        while True:
            frame, _recv_at, seq = STREAM.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq
            payload = _jpeg(frame, CFG.stream_width, CFG.stream_quality)
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            yield str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n"
            time.sleep(period)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/snapshot.jpg")
def api_snapshot():
    with _CALIB_LOCK:
        jpg = CALIB.get("_jpeg") if CALIB else None
    if jpg is None:
        return jsonify({"error": "no calibrated frame yet"}), 404
    return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/refimage/<name>")
def api_refimage(name: str):
    """The reference frame the online stage matched against (for eyeballing)."""
    if not ENGINE:
        return jsonify({"error": "engine not ready"}), 503
    if name not in {r.name for r in ENGINE.refs}:
        return jsonify({"error": "unknown reference"}), 404
    img = cv2.imread(os.path.join(CFG.ref_images, name))
    if img is None:
        return jsonify({"error": "cannot read reference image"}), 404
    return Response(_jpeg(img, 640, 75), mimetype="image/jpeg")


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    global CALIB, LAST_ATTEMPT

    if ENGINE is None:
        return jsonify({"ok": False, "reason": "engine not ready"}), 503

    frame, recv_at, seq = STREAM.latest()
    if frame is None:
        return jsonify({"ok": False, "reason": "no frame from the camera yet"}), 503

    t0 = time.time()
    result = ENGINE.calibrate(frame)
    result["total_s"] = round(time.time() - t0, 3)
    result["frame_seq"] = seq
    result["recv_at"] = recv_at.isoformat(timespec="milliseconds") if recv_at else None
    result["calibrated_at"] = datetime.now().isoformat(timespec="milliseconds")

    with _CALIB_LOCK:
        LAST_ATTEMPT = result
        if result["ok"]:
            result["_jpeg"] = _jpeg(frame, 0, 90)  # full resolution, for clicking
            result["_guard"] = _guard_signature(frame)
            CALIB = result
            _GUARD_CACHE.clear()

    return jsonify(_public_calib(result))


@app.route("/api/unproject", methods=["POST"])
def api_unproject():
    """Pixels on the calibrated frame -> world points on ``Z = plane_z``."""
    body = request.get_json(force=True, silent=True) or {}
    pts = body.get("points")
    if pts is None:
        if "u" not in body or "v" not in body:
            return jsonify({"error": "expected {u, v} or {points: [[u, v], ...]}"}), 400
        pts = [[body["u"], body["v"]]]
    try:
        uv = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return jsonify({"error": "points must be [[u, v], ...]"}), 400
    if uv.shape[0] == 0:
        return jsonify({"points": []})

    try:
        plane_z = float(body.get("plane_z", CFG.plane_z))
    except (TypeError, ValueError):
        return jsonify({"error": "plane_z must be a number"}), 400

    with _CALIB_LOCK:
        calib = CALIB
    if calib is None:
        return jsonify({"error": "not calibrated yet - press Capture & calibrate"}), 409

    # A click on the frozen frame is always answerable: that frame *is* the one
    # the calibration was solved on.  A click on the live view is only valid
    # while the camera still points where it did.
    drift = _view_drift()
    if body.get("source") == "live" and drift is not None and drift["stale"]:
        return (
            jsonify(
                {
                    "error": "the view moved since the calibration - re-calibrate",
                    "view_drift": drift,
                }
            ),
            409,
        )

    cam = calib["_camera"]
    xyz, valid = geometry.pixels_to_plane(cam, uv, plane_z)
    # round trip: project the recovered 3D point back and measure the pixel gap
    back, _in_front = geometry.points_to_pixels(cam, xyz)
    C = geometry.camera_center(cam)

    out = []
    for i in range(uv.shape[0]):
        item = {
            "pixel": [float(uv[i, 0]), float(uv[i, 1])],
            "valid": bool(valid[i]),
        }
        if valid[i]:
            item["world"] = [float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])]
            item["roundtrip_px"] = float(np.linalg.norm(back[i] - uv[i]))
            item["range_m"] = float(np.linalg.norm(xyz[i] - C))
            item["ground_range_m"] = float(np.linalg.norm(xyz[i, :2] - C[:2]))
            item["origin_range_m"] = float(np.linalg.norm(xyz[i]))
        else:
            item["reason"] = "the ray never meets Z = %.3f in front of the camera (above the horizon?)" % plane_z
        out.append(item)

    return jsonify(
        {
            "plane_z": plane_z,
            "camera_pos": C.tolist(),
            "frame_seq": calib.get("frame_seq"),
            "view_drift": drift,
            "points": out,
        }
    )


@app.route("/api/holdout", methods=["POST"])
def api_holdout():
    """Clicked pixels -> 2D->3D error in metres against the surveyed holdout points.

    The pixel must come from the user: projecting a survey point to get one and
    back-projecting it again is the identity and always reads 0.000 m.  See
    :mod:`live_web.holdout`.
    """
    body = request.get_json(force=True, silent=True) or {}
    pts = body.get("points") or []
    try:
        uv = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return jsonify({"error": "points must be [[u, v], ...]"}), 400

    names = body.get("targets")
    if names is not None and len(names) != uv.shape[0]:
        return jsonify({"error": "targets must be as long as points"}), 400
    if uv.shape[0] == 0:
        return jsonify({"points": [], "summary": None})

    with _CALIB_LOCK:
        calib = CALIB
    if calib is None:
        return jsonify({"error": "not calibrated yet - press Capture & calibrate"}), 409

    # same rule as /api/unproject: the frozen frame IS the calibrated one, the
    # live view only counts while the camera still points where it did
    drift = _view_drift()
    if body.get("source") == "live" and drift is not None and drift["stale"]:
        return (
            jsonify(
                {
                    "error": "the view moved since the calibration - re-calibrate",
                    "view_drift": drift,
                }
            ),
            409,
        )

    tgts = holdout.targets(ENGINE.survey if ENGINE else [])
    out = holdout.score(calib["_camera"], uv, tgts, names)
    out["view_drift"] = drift
    out["frame_seq"] = calib.get("frame_seq")
    return jsonify(out)


@app.route("/api/overlay")
def api_overlay():
    """Survey points + matched keypoints + horizon, in calibrated-frame pixels.

    Projecting known floor points forward is the independent check on the
    calibration: if the crosses land on the real markers, 2D->3D is sound.
    """
    with _CALIB_LOCK:
        calib = CALIB
    if calib is None:
        return jsonify({"error": "not calibrated yet"}), 409

    cam = calib["_camera"]
    w, h = calib["frame_size"]
    try:
        plane_z = float(request.args.get("plane_z", CFG.plane_z))
    except (TypeError, ValueError):
        return jsonify({"error": "plane_z must be a number"}), 400

    survey = []
    if ENGINE and ENGINE.survey:
        pts = np.asarray([s["xyz"] for s in ENGINE.survey], dtype=np.float64)
        uv, in_front = geometry.points_to_pixels(cam, pts)
        for i, s in enumerate(ENGINE.survey):
            if not in_front[i]:
                continue
            if not (-50 <= uv[i, 0] <= w + 50 and -50 <= uv[i, 1] <= h + 50):
                continue
            survey.append(
                {
                    "id": s["id"],
                    "split": s["split"],
                    "world": s["xyz"],
                    "pixel": [float(uv[i, 0]), float(uv[i, 1])],
                }
            )

    hz = geometry.horizon_v(cam, w)
    keypoints = calib.get("_matches_live_uv")
    return jsonify(
        {
            "frame_size": [w, h],
            "survey": survey,
            "horizon": hz.tolist() if hz is not None else None,
            "grid": _floor_grid(cam, w, h, plane_z),
            "grid_extent": _grid_extent(cam, w, h, plane_z),
            "matched_keypoints": (
                keypoints[:: max(1, len(keypoints) // 400)].tolist() if keypoints is not None else []
            ),
        }
    )


MAX_GRID_SPAN_M = 25.0


def _grid_extent(cam, width: int, height: int, plane_z: float):
    """World X/Y box the floor grid is drawn over: what the frame sees, clipped.

    The plane runs to infinity, so the visible footprint of a shallow view is
    hundreds of metres and the grid would cover walls, people and ceiling.  The
    surveyed area is the region anyone actually measures against, so the box is
    the visible footprint intersected with the survey bounding box (+2 m), and
    capped at ``MAX_GRID_SPAN_M`` when there is no survey.
    """
    us, vs = np.meshgrid(np.linspace(0, width - 1, 40), np.linspace(0, height - 1, 24))
    xyz, valid = geometry.pixels_to_plane(cam, np.stack([us.ravel(), vs.ravel()], axis=1), plane_z)
    if not np.any(valid):
        return None
    xy = xyz[valid][:, :2]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)

    if ENGINE and ENGINE.survey:
        s = np.asarray([p["xyz"][:2] for p in ENGINE.survey], dtype=np.float64)
        lo = np.maximum(lo, s.min(axis=0) - 2.0)
        hi = np.minimum(hi, s.max(axis=0) + 2.0)
    hi = np.minimum(hi, lo + MAX_GRID_SPAN_M)

    if not np.isfinite(np.concatenate([lo, hi])).all() or np.any(hi <= lo):
        return None
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


def _floor_grid(cam, width: int, height: int, plane_z: float, spacing: float = 1.0) -> list:
    """1 m floor grid over the surveyed part of the plane, as pixel polylines.

    Computed server-side with ``cv2.projectPoints`` so the lines carry the same
    distortion model as the click maths; the browser only draws them.
    """
    extent = _grid_extent(cam, width, height, plane_z)
    if extent is None:
        return []
    lo = np.array(extent[:2])
    hi = np.array(extent[2:])

    x0, x1 = np.floor(lo[0] / spacing) * spacing, np.ceil(hi[0] / spacing) * spacing
    y0, y1 = np.floor(lo[1] / spacing) * spacing, np.ceil(hi[1] / spacing) * spacing

    lines = []
    for axis, (a0, a1, b0, b1) in ((0, (x0, x1, y0, y1)), (1, (y0, y1, x0, x1))):
        n = int(round((a1 - a0) / spacing)) + 1
        if n > 120:
            continue
        for a in np.linspace(a0, a1, n):
            b = np.linspace(b0, b1, 120)
            pts = np.zeros((b.size, 3))
            pts[:, axis] = a
            pts[:, 1 - axis] = b
            pts[:, 2] = plane_z
            uv, in_front = geometry.points_to_pixels(cam, pts)
            keep = (
                in_front
                & (uv[:, 0] > -3 * width) & (uv[:, 0] < 4 * width)
                & (uv[:, 1] > -3 * height) & (uv[:, 1] < 4 * height)
            )
            if np.count_nonzero(keep) < 2:
                continue
            lines.append(
                {
                    "axis": "x" if axis == 0 else "y",
                    "value": float(a),
                    "major": bool(abs(a) < 1e-9),
                    "pixels": np.round(uv[keep], 1).tolist(),
                }
            )
    return lines


# --------------------------------------------------------------------------
# PTZ control (ONVIF)
# --------------------------------------------------------------------------
# The UI polls the pose, so both reads are cached: several browser tabs must not
# turn into several SOAP round trips per second.  Presets only change when a
# human stores one, so they are cached far longer.
_PTZ_LOCK = threading.Lock()
_PTZ_CACHE: dict = {}
_PTZ_STATUS_TTL = 0.8
_PTZ_PRESET_TTL = 60.0


def _ptz_cached(key: str, ttl: float, produce):
    """``produce()`` at most once per ``ttl`` seconds, shared across requests."""
    with _PTZ_LOCK:
        hit = _PTZ_CACHE.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    value = produce()
    with _PTZ_LOCK:
        _PTZ_CACHE[key] = (time.time(), value)
    return value


def _ptz_invalidate() -> None:
    with _PTZ_LOCK:
        _PTZ_CACHE.pop("status", None)


def _ptz_ready():
    """The camera, or a ready-to-return error response."""
    if PTZ is None:
        return None, (jsonify({"available": False, "reason": "PTZ control is disabled"}), 503)
    if not PTZ.ready:
        # the camera may have come back since startup - one cheap retry
        PTZ.connect()
    if not PTZ.ready:
        return None, (jsonify({"available": False, "reason": PTZ.last_error}), 503)
    return PTZ, None


def _ptz_axis(body: dict, key: str, human_to_norm, norm_range) -> Optional[float]:
    """One axis of a move command, or ``None`` to leave that axis alone.

    ``units: "deg"`` means degrees for pan/tilt and magnification for zoom;
    ``"norm"`` is the raw ONVIF space the camera actually consumes.
    """
    if body.get(key) is None:
        return None
    try:
        value = float(body[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number")
    if body.get("units", "deg") != "norm":
        value = human_to_norm(value)
    lo, hi = norm_range
    if not (lo - 1e-9 <= value <= hi + 1e-9):
        raise ValueError(f"{key} is outside the range the camera accepts")
    return value


@app.route("/api/ptz")
def api_ptz():
    """Current pose and the preset list, in both wire units and degrees."""
    cam, err = _ptz_ready()
    if err:
        return err
    try:
        payload = {
            "available": True,
            "device": cam.device,
            "profile": cam.profile_token,
            "units": ptz.unit_ranges(),
            "position": _ptz_cached("status", _PTZ_STATUS_TTL, cam.status),
        }
        if request.args.get("presets") != "0":
            payload["presets"] = _ptz_cached("presets", _PTZ_PRESET_TTL, cam.presets)
    except ptz.PtzError as exc:
        return jsonify({"available": False, "reason": str(exc)}), 502
    return jsonify(payload)


@app.route("/api/ptz/move", methods=["POST"])
def api_ptz_move():
    """Absolute move.  Axes left out (or null) keep their current value.

    The calibration is *not* cleared here: the view guard notices the movement
    on its own and, with "auto on move" on, the app re-solves once the camera
    settles.  Doing it here as well would race that path.
    """
    cam, err = _ptz_ready()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    try:
        pan = _ptz_axis(body, "pan", ptz.pan_from_deg, (-1.0, 1.0))
        tilt = _ptz_axis(body, "tilt", ptz.tilt_from_deg, (-1.0, 1.0))
        zoom = _ptz_axis(body, "zoom", ptz.zoom_from_x, (0.0, 1.0))
    except ValueError as exc:
        return jsonify({"ok": False, "reason": str(exc)}), 400
    if pan is None and tilt is None and zoom is None:
        return jsonify({"ok": False, "reason": "no axis given"}), 400

    speed = body.get("speed", CFG.ptz_speed)
    try:
        target = cam.absolute_move(pan, tilt, zoom, speed)
    except ptz.PtzError as exc:
        return jsonify({"ok": False, "reason": str(exc)}), 502
    _ptz_invalidate()
    pan_deg, tilt_deg, zoom_x = ptz.norm_to_deg(target["pan"], target["tilt"], target["zoom"])
    return jsonify(
        {
            "ok": True,
            "target": dict(target, pan_deg=pan_deg, tilt_deg=tilt_deg, zoom_x=zoom_x),
        }
    )


@app.route("/api/ptz/preset", methods=["POST"])
def api_ptz_preset():
    """Recall a preset by token.

    Tokens 65-99 are not poses on this firmware - they are vendor commands
    ("open menu", "auto scan", "turn on ir led").  Recalling one by a slip of
    the finger would do something the operator did not ask for, so they need an
    explicit ``force``.
    """
    cam, err = _ptz_ready()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    token = str(body.get("token", "")).strip()
    if not token:
        return jsonify({"ok": False, "reason": "no preset token given"}), 400

    try:
        presets = _ptz_cached("presets", _PTZ_PRESET_TTL, cam.presets)
    except ptz.PtzError as exc:
        return jsonify({"ok": False, "reason": str(exc)}), 502
    known = {p["token"]: p for p in presets}
    if token not in known:
        return jsonify({"ok": False, "reason": f"preset {token} does not exist"}), 400
    preset = known[token]
    if preset["kind"] == "reserved" and not body.get("force"):
        return jsonify(
            {
                "ok": False,
                "reason": f"preset {token} ('{preset['name']}') is a vendor command, "
                          "not a camera pose - pass force to send it anyway",
            }
        ), 409

    try:
        cam.goto_preset(token, body.get("speed", CFG.ptz_speed))
    except ptz.PtzError as exc:
        return jsonify({"ok": False, "reason": str(exc)}), 502
    _ptz_invalidate()
    return jsonify({"ok": True, "preset": preset})


@app.route("/api/ptz/stop", methods=["POST"])
def api_ptz_stop():
    cam, err = _ptz_ready()
    if err:
        return err
    try:
        cam.stop()
    except ptz.PtzError as exc:
        return jsonify({"ok": False, "reason": str(exc)}), 502
    _ptz_invalidate()
    return jsonify({"ok": True})


@app.route("/api/config")
def api_config():
    return jsonify(dataclasses.asdict(CFG))


# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref_images", default=d.ref_images, help="registered reference images (symlink dir)")
    p.add_argument("--ref_features", default=d.ref_features, help="offline COLMAP features")
    p.add_argument("--ref_params", default=d.ref_params, help="georeferenced offline camera JSON")
    p.add_argument("--mask", default=d.mask, help="keypoint mask, non-zero = keep ('' to disable)")
    p.add_argument("--survey_points", default=d.survey_points)
    p.add_argument("--rtsp_url", default="", help="overrides the URL from .env")
    p.add_argument("--env_file", default=d.env_file)
    p.add_argument("--no_dist", action="store_true", help="use FactorType.F instead of FDist")
    p.add_argument("--plane_z", type=float, default=d.plane_z, help="world Z of the click plane (m)")
    p.add_argument(
        "--drift_tolerance_px",
        type=float,
        default=d.drift_tolerance_px,
        help="live view may drift this far (px) before the calibration counts as stale",
    )
    p.add_argument("--no_ptz", action="store_true", help="do not connect to the camera's ONVIF PTZ service")
    p.add_argument("--ptz_profile", default=d.ptz_profile, help="ONVIF media profile token (default: first with PTZ)")
    p.add_argument(
        "--ptz_speed", type=float, default=d.ptz_speed,
        help="default move speed, 0..1 in the ONVIF generic speed space",
    )
    p.add_argument("--device", default="", help="cuda / cuda:1 / cpu (default: auto)")
    p.add_argument("--min_inliers", type=int, default=d.min_inliers)
    p.add_argument("--conf_thresh", type=float, default=d.conf_thresh)
    p.add_argument("--host", default=d.host)
    p.add_argument("--port", type=int, default=d.port)
    p.add_argument("--stream_width", type=int, default=d.stream_width)
    p.add_argument("--offline_frame", default="", help="debug: calibrate this image instead of RTSP and exit")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(
        ref_images=args.ref_images,
        ref_features=args.ref_features,
        ref_params=args.ref_params,
        mask=args.mask,
        survey_points=args.survey_points,
        env_file=args.env_file,
        dist=not args.no_dist,
        device=args.device,
        conf_thresh=args.conf_thresh,
        min_inliers=args.min_inliers,
        plane_z=args.plane_z,
        drift_tolerance_px=args.drift_tolerance_px,
        ptz_enabled=not args.no_ptz,
        ptz_profile=args.ptz_profile,
        ptz_speed=args.ptz_speed,
        host=args.host,
        port=args.port,
        stream_width=args.stream_width,
    )
    cfg.rtsp_url = args.rtsp_url or load_rtsp_url(cfg.env_file)
    return cfg


def _port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("" if host == "0.0.0.0" else host, port))
        except OSError:
            return True
    return False


def main(argv=None) -> int:
    global CFG, ENGINE, PTZ, STREAM

    args = parse_args(argv)
    CFG = build_config(args)
    ENGINE = RelocEngine(CFG)

    if args.offline_frame:
        frame = cv2.imread(args.offline_frame)
        if frame is None:
            print(f"cannot read {args.offline_frame}", file=sys.stderr)
            return 1
        result = ENGINE.calibrate(frame)
        print(json.dumps(_public_calib(result), indent=2))
        if result["ok"]:
            cam = result["_camera"]
            h, w = frame.shape[:2]
            uv = np.array([[w / 2.0, h * 0.75]])
            xyz, valid = geometry.pixels_to_plane(cam, uv, CFG.plane_z)
            print(f"centre-lower pixel {uv[0].tolist()} -> {xyz[0].tolist()} valid={bool(valid[0])}")
        return 0 if result["ok"] else 2

    if _port_in_use(CFG.host, CFG.port):
        print(f"[app] port {CFG.port} is already in use - pass --port <other>", file=sys.stderr)
        return 1

    if CFG.ptz_enabled:
        # A camera that answers RTSP but not ONVIF is a working setup minus one
        # feature, so a failure here only disables the PTZ card.
        ptz_cfg = ptz.load_ptz_config(CFG.env_file, CFG.rtsp_url)
        ptz_cfg.profile_token = CFG.ptz_profile
        PTZ = ptz.OnvifPtz(ptz_cfg).connect()

    STREAM = CameraStream(CFG.rtsp_url).start()
    print(f"[app] waiting for the first frame from {CFG.rtsp_url.split('@')[-1]} ...")
    if not STREAM.wait_for_frame(20.0):
        print("[app] no frame yet - the UI will keep retrying in the background")

    print(f"[app] http://{'localhost' if CFG.host == '0.0.0.0' else CFG.host}:{CFG.port}")
    app.run(host=CFG.host, port=CFG.port, threaded=True, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
