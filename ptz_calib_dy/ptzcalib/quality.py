"""Calibration quality grading, shared by the offline evaluator and the live app.

Unlike the rest of this package this module has **no C++ counterpart**: PTZ-Calib
reports raw errors and leaves the pass/fail call to the caller.  What is graded
here is the project KPI - *multi-camera position registration error*, in metres,
with a limit of 0.8 m - measured as the holdout 2D-3D metre RMSE
(``scripts/eval_own.py::holdout_2d3d``).

Six states exist because a live viewer needs more than a score.  Only the four
*measurable* ones can come out of a batch evaluation, and ``classify()`` returns
exactly those:

``UNKNOWN``   no usable RMSE, so no judgement can be made yet
``VALID``     ``rmse <= pass_threshold_m`` (0.8 m)
``DEGRADED``  ``pass_threshold_m < rmse <= degraded_threshold_m`` (1.0 m) - usable with care
``INVALID``   ``rmse > degraded_threshold_m``

The remaining two describe *which* calibration is in use rather than how accurate
it is.  They are runtime observations, so ``classify()`` never returns them and
``eval_own.py`` never emits them; ``live_web`` sets them directly:

``FALLBACK``  a previous or substitute calibration is being used
``MOVING``    the camera is moving (PTZ), so the current solution is untrustworthy

A missing RMSE grades ``UNKNOWN``, never ``INVALID``: no georeferencing, no
holdout survey, or no surviving observation all mean nothing was *measured*, and
an unmeasured calibration has not failed - it is simply unjudged.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

# The project KPI: multi-camera position registration error <= 0.8 m.
DEFAULT_PASS_THRESHOLD_M = 0.8
# Upper limit of the "usable with care" band above the KPI.
DEFAULT_DEGRADED_THRESHOLD_M = 1.0


class CalibrationQuality(str, Enum):
    """Grade of one calibration.  ``str`` mixin so ``.value`` round-trips through JSON."""

    UNKNOWN = "unknown"
    VALID = "valid"
    DEGRADED = "degraded"
    INVALID = "invalid"
    FALLBACK = "fallback"
    MOVING = "moving"

    def __str__(self) -> str:  # keeps f-strings readable across Python versions
        return self.value


#: The subset ``classify()`` can return - the states a measurement can establish.
MEASURABLE = (
    CalibrationQuality.UNKNOWN,
    CalibrationQuality.VALID,
    CalibrationQuality.DEGRADED,
    CalibrationQuality.INVALID,
)

#: The subset only a running system can establish; no evaluation produces these.
RUNTIME_ONLY = (CalibrationQuality.FALLBACK, CalibrationQuality.MOVING)


def classify(
    rmse_m: Optional[float],
    pass_threshold_m: float = DEFAULT_PASS_THRESHOLD_M,
    degraded_threshold_m: float = DEFAULT_DEGRADED_THRESHOLD_M,
) -> CalibrationQuality:
    """Grade one metre RMSE.  ``None``/NaN -> ``UNKNOWN`` (see the module docstring).

    Both bounds are inclusive on their upper side, so a run sitting exactly on
    the KPI (``rmse == 0.8``) is ``VALID`` and one sitting exactly on the limit
    (``rmse == 1.0``) is ``DEGRADED``.
    """
    if rmse_m is None:
        return CalibrationQuality.UNKNOWN
    value = float(rmse_m)
    if math.isnan(value):
        return CalibrationQuality.UNKNOWN
    if value <= pass_threshold_m:
        return CalibrationQuality.VALID
    if value <= degraded_threshold_m:
        return CalibrationQuality.DEGRADED
    return CalibrationQuality.INVALID
