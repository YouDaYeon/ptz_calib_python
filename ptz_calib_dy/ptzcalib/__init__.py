"""Python re-implementation of PTZ-Calib (ICRA 2025).

Module layout mirrors the public C++ sources:

===========================  =================================================
``types``                    ``src/core/types.{h,cc}``
``data_io``                  ``src/core/data_io.{h,cc}``
``union_find`` / ``tracks``  ``src/core/union_find.h`` / ``src/core/tracks.{h,cc}``
``krt_optimizer``            ``src/core/krt_optimizer.{h,cc}``
``ptzray_optimizer``         ``src/core/ptzray_optimizer.{h,cc}``
``ptz_incremental_optimizer````src/core/ptz_incremental_optimizer.{h,cc}``
``lm``                       the ceres LM/trust-region machinery those use
``os_path`` / ``logging_utils`` ``src/utils/``
``quality``                  (no C++ counterpart; calibration quality grading)
===========================  =================================================
"""

from .types import Camera, ImageFeatures, MatchesInfo, Ray  # noqa: F401

__all__ = ["Camera", "ImageFeatures", "MatchesInfo", "Ray"]
