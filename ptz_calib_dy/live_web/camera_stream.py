"""Background RTSP grabber.

RTSP must be drained continuously or the decoder backs up and every read
returns an ever-older frame, so a thread keeps only the newest one.  The web
handlers never touch ``cv2.VideoCapture`` directly.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def load_rtsp_url(env_file: str) -> str:
    """Same resolution order as ``camera_test.py``: ``URL`` first, else parts."""
    from dotenv import load_dotenv

    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=True)

    def get(key: str, default: str = "") -> str:
        return os.getenv(key, default).strip().strip("'\"")

    url = get("URL")
    if url:
        return url

    user, password = get("USER"), get("PASSWORD")
    if not user or not password:
        raise ValueError(f"{env_file}: need URL, or USER and PASSWORD")
    host = get("HOST", "10.0.110.241")
    port = get("PORT", "554")
    channel = get("CHANNEL", "stream0")
    return f"rtsp://{user}:{password}@{host}:{port}/{channel}"


class CameraStream:
    """Latest-frame-wins RTSP reader with automatic reconnect."""

    def __init__(self, url: str, log=print, reconnect_delay: float = 3.0) -> None:
        self.url = url
        self.log = log
        self.reconnect_delay = reconnect_delay

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._recv_at: Optional[datetime] = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.connected = False
        self.last_error = ""
        self.fps = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "CameraStream":
        self._thread = threading.Thread(target=self._run, name="rtsp-grabber", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def wait_for_frame(self, timeout: float = 15.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._frame is not None:
                return True
            if self._stop.is_set():
                return False
            time.sleep(0.05)
        return False

    # -- accessors ---------------------------------------------------------
    def latest(self) -> Tuple[Optional[np.ndarray], Optional[datetime], int]:
        """A *copy* of the newest frame, its receive time, and its sequence id."""
        with self._lock:
            if self._frame is None:
                return None, None, self._seq
            return self._frame.copy(), self._recv_at, self._seq

    # -- worker ------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # backend may not support it
                pass

            if not cap.isOpened():
                self.connected = False
                self.last_error = "cannot open RTSP stream"
                self.log(f"[rtsp] open failed, retrying in {self.reconnect_delay:.0f}s")
                cap.release()
                self._stop.wait(self.reconnect_delay)
                continue

            self.connected = True
            self.last_error = ""
            self.log(f"[rtsp] connected: {self.url.split('@')[-1]}")

            n, t_window = 0, time.time()
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.last_error = "frame read failed"
                    break
                with self._lock:
                    self._frame = frame
                    self._recv_at = datetime.now()
                    self._seq += 1
                n += 1
                dt = time.time() - t_window
                if dt >= 2.0:
                    self.fps = n / dt
                    n, t_window = 0, time.time()

            self.connected = False
            cap.release()
            if not self._stop.is_set():
                self.log(f"[rtsp] disconnected ({self.last_error}), reconnecting")
                self._stop.wait(self.reconnect_delay)
