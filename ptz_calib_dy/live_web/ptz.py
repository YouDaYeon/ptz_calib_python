"""ONVIF PTZ control - absolute pan/tilt/zoom moves and preset recalls.

Written against the stdlib only (``urllib`` + ``xml.etree``).  ``.venv-match``
carries torch, OpenCV and Flask; adding ``onvif-zeep``/``zeep`` for what amounts
to six SOAP calls would be a large dependency for a small surface.

Units
-----
This camera (``Security Dev TC-H354S SPEC:23X``) mixes two conventions and the
mix is not a bug on our side, so it is handled here once:

* ``GetStatus`` reports the **ONVIF generic space** - pan/tilt in ``[-1, 1]``,
  zoom in ``[0, 1]``.  ``AbsoluteMove`` consumes the same space.  Those numbers
  are exact and are what travels over the wire.
* ``GetPresets`` reports **degrees and optical magnification** (``pan 210.9``,
  ``tilt 9.4``, ``zoom 1.7``) while claiming the same space URI.

``norm_to_deg`` / ``deg_to_norm`` convert between them.  The constants were
derived from the camera itself, not from the ONVIF spec - see ``_UNITS``.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# unit conversion
# --------------------------------------------------------------------------
# pan:  (x + 1) * 180        - normalised 0.171667 reads 210.9 deg, which is
#                              exactly the pan stored in presets 3 and 4.
# zoom: 1 + 22 * z           - normalised 0.031818 reads 1.7x, exactly the zoom
#                              stored in presets 1-4, and z=1 gives the 23x the
#                              model name claims.
# tilt: see TILT_A / TILT_B  - measured, because no preset row pinned it down.
PAN_DEG_MIN, PAN_DEG_MAX = 0.0, 360.0
ZOOM_X_MIN, ZOOM_X_MAX = 1.0, 23.0

# tilt: measured, because no preset row pinned it down by arithmetic alone.
# Drove the camera to presets 3, 4 and 11 (tilt 9.4 / 16.4 / 35.0 deg per the
# preset table) and read back the normalised y - 0.733333 / 0.583011 / 0.182796.
# All three two-point fits agree to 0.05 in both coefficients, so the map is
# linear, and the endpoints land on round numbers: y=+1 is 3 deg above the
# horizon, y=-1 is straight down.
TILT_DEG_AT_UP, TILT_DEG_AT_DOWN = -3.0, 90.0    # y = +1 and y = -1
TILT_A = 0.5 * (TILT_DEG_AT_DOWN + TILT_DEG_AT_UP)    #  43.5
TILT_B = 0.5 * (TILT_DEG_AT_UP - TILT_DEG_AT_DOWN)    # -46.5


# per axis, because a command may name only one of them
def pan_to_deg(x: float) -> float:
    return (x + 1.0) * 180.0


def pan_from_deg(deg: float) -> float:
    return deg / 180.0 - 1.0


def tilt_to_deg(y: float) -> float:
    return TILT_A + TILT_B * y


def tilt_from_deg(deg: float) -> float:
    return (deg - TILT_A) / TILT_B


def zoom_to_x(z: float) -> float:
    return ZOOM_X_MIN + (ZOOM_X_MAX - ZOOM_X_MIN) * z


def zoom_from_x(x: float) -> float:
    return (x - ZOOM_X_MIN) / (ZOOM_X_MAX - ZOOM_X_MIN)


def norm_to_deg(pan: float, tilt: float, zoom: float) -> Tuple[float, float, float]:
    """ONVIF generic space -> (pan deg, tilt deg, zoom x)."""
    return pan_to_deg(pan), tilt_to_deg(tilt), zoom_to_x(zoom)


def deg_to_norm(pan_deg: float, tilt_deg: float, zoom_x: float) -> Tuple[float, float, float]:
    """(pan deg, tilt deg, zoom x) -> ONVIF generic space."""
    return pan_from_deg(pan_deg), tilt_from_deg(tilt_deg), zoom_from_x(zoom_x)


def unit_ranges() -> dict:
    """Input limits for the UI, so the numbers live in exactly one place."""
    return {
        "pan": {"norm": [-1.0, 1.0], "human": [PAN_DEG_MIN, PAN_DEG_MAX], "unit": "\u00b0"},
        "tilt": {"norm": [-1.0, 1.0], "human": [TILT_DEG_AT_UP, TILT_DEG_AT_DOWN], "unit": "\u00b0"},
        "zoom": {"norm": [0.0, 1.0], "human": [ZOOM_X_MIN, ZOOM_X_MAX], "unit": "\u00d7"},
    }


def clamp_norm(pan: float, tilt: float, zoom: float) -> Tuple[float, float, float]:
    """Keep a command inside the spaces the node advertises, silently."""
    c = lambda v, lo, hi: max(lo, min(hi, float(v)))  # noqa: E731
    return c(pan, -1.0, 1.0), c(tilt, -1.0, 1.0), c(zoom, 0.0, 1.0)


# --------------------------------------------------------------------------
class PtzError(RuntimeError):
    """A SOAP fault, an HTTP error, or an unreachable camera."""


@dataclass
class PtzConfig:
    host: str = ""
    port: int = 80
    user: str = ""
    password: str = ""
    profile_token: str = ""   # "" -> first profile that has a PTZ configuration
    timeout: float = 6.0


def load_ptz_config(env_file: str, rtsp_url: str = "") -> PtzConfig:
    """Credentials from ``.env``, host from ``HOST`` or the RTSP URL.

    ``.env`` in this deployment sets ``URL`` (which carries user, password and
    host) plus ``ONVIF_PORT``, and leaves ``HOST`` empty - so the RTSP URL is
    the fallback for all three.
    """
    from dotenv import load_dotenv

    if env_file and os.path.exists(env_file):
        load_dotenv(env_file, override=True)

    def get(key: str, default: str = "") -> str:
        return os.getenv(key, default).strip().strip("'\"")

    parsed = urlparse(rtsp_url or get("URL"))
    return PtzConfig(
        host=get("HOST") or (parsed.hostname or ""),
        port=int(get("ONVIF_PORT", "80") or 80),
        user=get("USER") or (parsed.username or ""),
        password=get("PASSWORD") or (parsed.password or ""),
        profile_token=get("ONVIF_PROFILE"),
    )


# --------------------------------------------------------------------------
_NS = (
    'xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
    'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
    'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
    'xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" '
    'xmlns:tt="http://www.onvif.org/ver10/schema"'
)
_WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
_WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
_PW_DIGEST = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
)
_B64 = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-soap-message-security-1.0#Base64Binary"
)

# The band this firmware reserves for vendor commands - the names in it are
# "open menu", "auto scan", "turn on ir led" and so on, not camera poses.
# Sending one of those as a preset recall would trigger the command.
_RESERVED_PRESETS = range(65, 100)
_PLACEHOLDER_NAME = re.compile(r"^(?:PRESET_?\d+|\d+)$", re.I)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class OnvifPtz:
    """One camera's PTZ service.  Thread-safe; commands are serialised."""

    def __init__(self, cfg: PtzConfig, log=print) -> None:
        self.cfg = cfg
        self.log = log
        self._lock = threading.Lock()
        self._base = f"http://{cfg.host}:{cfg.port}"
        self._svc: Dict[str, str] = {
            "device": f"{self._base}/onvif/device_service",
            "media": f"{self._base}/onvif/media",
            "ptz": f"{self._base}/onvif/ptz",
        }
        self.profile_token: str = cfg.profile_token
        self.device: Dict[str, str] = {}
        self.ready = False
        self.last_error = ""

    # -- SOAP --------------------------------------------------------------
    def _security_header(self) -> str:
        nonce = os.urandom(16)
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        digest = base64.b64encode(
            hashlib.sha1(nonce + created.encode() + self.cfg.password.encode()).digest()
        ).decode()
        return (
            f'<s:Header><Security s:mustUnderstand="1" xmlns="{_WSSE}"><UsernameToken>'
            f"<Username>{self.cfg.user}</Username>"
            f'<Password Type="{_PW_DIGEST}">{digest}</Password>'
            f'<Nonce EncodingType="{_B64}">{base64.b64encode(nonce).decode()}</Nonce>'
            f'<Created xmlns="{_WSU}">{created}</Created>'
            "</UsernameToken></Security></s:Header>"
        )

    def _call(self, service: str, body: str) -> ET.Element:
        """POST one SOAP body, return the response ``Body`` element."""
        env = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<s:Envelope {_NS}>{self._security_header()}<s:Body>{body}</s:Body></s:Envelope>"
        )
        req = urllib.request.Request(
            self._svc[service],
            data=env.encode("utf-8"),
            headers={"Content-Type": "application/soap+xml; charset=utf-8"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:          # a fault still has a body
            raw = exc.read()
            reason = _fault_reason(raw)
            raise PtzError(reason or f"HTTP {exc.code} from {service}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise PtzError(f"{self.cfg.host}:{self.cfg.port} unreachable ({exc})") from None

        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise PtzError(f"malformed SOAP response ({exc})") from None
        reason = _fault_reason(raw)
        if reason:
            raise PtzError(reason)
        for child in root:
            if _local(child.tag) == "Body":
                return child
        raise PtzError("SOAP response has no Body")

    # -- discovery ---------------------------------------------------------
    def connect(self) -> "OnvifPtz":
        """Read device info, resolve the service addresses, pick a profile.

        The XAddr values come from the camera rather than being hardcoded, so a
        firmware that moves ``/onvif/ptz`` elsewhere keeps working.
        """
        with self._lock:
            try:
                info = self._call("device", "<tds:GetDeviceInformation/>")
                self.device = {
                    _local(e.tag).lower(): (e.text or "").strip()
                    for e in list(info)[0]
                }

                caps = self._call(
                    "device",
                    "<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>",
                )
                for node in caps.iter():
                    name = _local(node.tag)
                    if name in ("PTZ", "Media"):
                        for child in node:
                            if _local(child.tag) == "XAddr" and child.text:
                                self._svc[name.lower()] = child.text.strip()

                if not self.profile_token:
                    self.profile_token = self._first_ptz_profile()

                self.ready = True
                self.last_error = ""
                self.log(
                    f"[ptz] {self.device.get('manufacturer', '?')} "
                    f"{self.device.get('model', '?')} · profile {self.profile_token}"
                )
            except PtzError as exc:
                self.ready = False
                self.last_error = str(exc)
                self.log(f"[ptz] unavailable: {exc}")
        return self

    def _first_ptz_profile(self) -> str:
        body = self._call("media", "<trt:GetProfiles/>")
        first = ""
        for prof in body.iter():
            if _local(prof.tag) != "Profiles":
                continue
            token = prof.attrib.get("token", "")
            first = first or token
            if any(_local(c.tag) == "PTZConfiguration" for c in prof):
                return token
        if not first:
            raise PtzError("camera reports no media profiles")
        return first

    # -- read --------------------------------------------------------------
    def status(self) -> dict:
        """Current pose, in both the wire units and degrees/magnification."""
        with self._lock:
            body = self._call(
                "ptz",
                f"<tptz:GetStatus><tptz:ProfileToken>{self.profile_token}"
                "</tptz:ProfileToken></tptz:GetStatus>",
            )
        pan = tilt = zoom = 0.0
        moving = {}
        utc = ""
        for node in body.iter():
            name = _local(node.tag)
            if name == "PanTilt" and "x" in node.attrib:
                pan, tilt = float(node.attrib["x"]), float(node.attrib["y"])
            elif name == "Zoom" and "x" in node.attrib:
                zoom = float(node.attrib["x"])
            elif name in ("PanTilt", "Zoom") and node.text:
                moving[name.lower()] = node.text.strip()
            elif name == "UtcTime":
                utc = (node.text or "").strip()
        pan_deg, tilt_deg, zoom_x = norm_to_deg(pan, tilt, zoom)
        return {
            "pan": pan, "tilt": tilt, "zoom": zoom,
            "pan_deg": pan_deg, "tilt_deg": tilt_deg, "zoom_x": zoom_x,
            "moving": bool(moving) and any(v.upper() != "IDLE" for v in moving.values()),
            "move_status": moving,
            "utc": utc,
        }

    def presets(self) -> List[dict]:
        """Every preset slot, tagged so the UI can hide the ones that are traps.

        The camera returns all 256 slots.  Only a few hold a pose a human set;
        65-99 are vendor commands ("open menu", "auto scan") that would *do*
        something rather than move, and the rest are unset slots the firmware
        fills with the sentinel ``1.0 / 1.0 / 1.0``.
        """
        with self._lock:
            body = self._call(
                "ptz",
                f"<tptz:GetPresets><tptz:ProfileToken>{self.profile_token}"
                "</tptz:ProfileToken></tptz:GetPresets>",
            )
        out: List[dict] = []
        for node in body.iter():
            if _local(node.tag) != "Preset":
                continue
            token = node.attrib.get("token", "")
            name, pt, zm = "", {}, {}
            for e in node.iter():
                tag = _local(e.tag)
                if tag == "Name":
                    name = (e.text or "").strip()
                elif tag == "PanTilt":
                    pt = e.attrib
                elif tag == "Zoom":
                    zm = e.attrib
            pan_deg = _f(pt.get("x"))
            tilt_deg = _f(pt.get("y"))
            zoom_x = _f(zm.get("x"))

            kind = "user"
            if token.isdigit() and int(token) in _RESERVED_PRESETS:
                kind = "reserved"
            elif (pan_deg, tilt_deg, zoom_x) == (1.0, 1.0, 1.0) and _PLACEHOLDER_NAME.match(name):
                kind = "empty"
            out.append(
                {
                    "token": token, "name": name, "kind": kind,
                    "pan_deg": pan_deg, "tilt_deg": tilt_deg, "zoom_x": zoom_x,
                }
            )
        return out

    # -- move --------------------------------------------------------------
    def absolute_move(
        self,
        pan: Optional[float] = None,
        tilt: Optional[float] = None,
        zoom: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> dict:
        """Move to an absolute pose, in the ONVIF generic space.

        Axes left as ``None`` are filled from the current pose: ONVIF has no
        "keep this axis" marker, and omitting ``PanTilt`` entirely makes some
        firmwares reset it.  Reading first costs one extra round trip and makes
        "zoom only" mean what it says.
        """
        now = self.status()
        pan = now["pan"] if pan is None else pan
        tilt = now["tilt"] if tilt is None else tilt
        zoom = now["zoom"] if zoom is None else zoom
        pan, tilt, zoom = clamp_norm(pan, tilt, zoom)

        speed_xml = ""
        if speed is not None:
            s = max(0.0, min(1.0, float(speed)))
            speed_xml = (
                f'<tptz:Speed><tt:PanTilt x="{s:.4f}" y="{s:.4f}"/>'
                f'<tt:Zoom x="{s:.4f}"/></tptz:Speed>'
            )
        with self._lock:
            self._call(
                "ptz",
                f"<tptz:AbsoluteMove><tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>"
                f'<tptz:Position><tt:PanTilt x="{pan:.6f}" y="{tilt:.6f}"/>'
                f'<tt:Zoom x="{zoom:.6f}"/></tptz:Position>{speed_xml}</tptz:AbsoluteMove>',
            )
        return {"pan": pan, "tilt": tilt, "zoom": zoom}

    def goto_preset(self, token: str, speed: Optional[float] = None) -> None:
        speed_xml = ""
        if speed is not None:
            s = max(0.0, min(1.0, float(speed)))
            speed_xml = (
                f'<tptz:Speed><tt:PanTilt x="{s:.4f}" y="{s:.4f}"/>'
                f'<tt:Zoom x="{s:.4f}"/></tptz:Speed>'
            )
        with self._lock:
            self._call(
                "ptz",
                f"<tptz:GotoPreset><tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>"
                f"<tptz:PresetToken>{token}</tptz:PresetToken>{speed_xml}</tptz:GotoPreset>",
            )

    def stop(self) -> None:
        with self._lock:
            self._call(
                "ptz",
                f"<tptz:Stop><tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>"
                "<tptz:PanTilt>true</tptz:PanTilt><tptz:Zoom>true</tptz:Zoom></tptz:Stop>",
            )

    # -- helpers -----------------------------------------------------------
    def wait_until_idle(self, timeout: float = 20.0, poll: float = 0.4) -> dict:
        """Block until the camera reports IDLE on both axes (or ``timeout``)."""
        import time

        deadline = time.time() + timeout
        st = self.status()
        while st["moving"] and time.time() < deadline:
            time.sleep(poll)
            st = self.status()
        return st


def _f(text: Optional[str]) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _fault_reason(raw: bytes) -> str:
    """Pull the human-readable text out of a SOAP fault, if this is one."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return ""
    if not any(_local(e.tag) == "Fault" for e in root.iter()):
        return ""
    parts = [
        (e.text or "").strip()
        for e in root.iter()
        if _local(e.tag) in ("Text", "Value") and (e.text or "").strip()
    ]
    return " / ".join(dict.fromkeys(parts)) or "SOAP fault"
