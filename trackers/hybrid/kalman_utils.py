from __future__ import annotations
"""kalman_utils.py
Wrapper tiện lợi để dùng KalmanFilter_box cho HybridTracker.
Nếu thư viện gốc tại PoseTrack không khả dụng dưới PYTHONPATH, ta cung cấp
phiên bản rút gọn đủ cho nhu cầu dự đoán bbox (giữ last measurement).
"""
# pylint: disable=missing-class-docstring, too-few-public-methods

from typing import Optional, Sequence
import importlib
import importlib.util
import os
import sys
import numpy as np

__all__ = ["KalmanFilter_box"]

_KF = None

# Thử import implement gốc
try:
    from PoseTrack.track.Tracker.kalman_filter_box_zya import KalmanFilter_box as _KF  # type: ignore
except ModuleNotFoundError:
    # Thử load bằng đường dẫn tuyệt đối nếu tồn tại
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PoseTrack/track/Tracker/kalman_filter_box_zya.py"))
    if os.path.isfile(_root):
        spec = importlib.util.spec_from_file_location("kalman_filter_box_zya", _root)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # type: ignore[arg-type]
            spec.loader.exec_module(module)  # type: ignore[arg-type]
            _KF = getattr(module, "KalmanFilter_box", None)

if _KF is None:

    class _SimpleKF:  # pragma: no cover
        """Phiên bản giản lược – chỉ lưu measurement cuối và trả về nó khi project."""

        def __init__(self):
            self.mean: Optional[np.ndarray] = None  # l, t, r, b
            self.covariance = None

        # API tương thích: update(measurement)
        def update(self, measurement: Sequence[float]):
            self.mean = np.asarray(measurement, dtype=float)

        def project(self):
            return self.mean.copy() if self.mean is not None else None

        # Vectorised predict placeholder (không dùng trong bản giản lược)
        def multi_predict(self, mean, covariance):
            return mean, covariance

    _KF = _SimpleKF

# Alias ra bên ngoài
KalmanFilter_box = _KF  # type: ignore 