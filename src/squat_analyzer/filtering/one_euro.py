"""
One-Euro Filter Implementation
==============================

The One-Euro Filter is an adaptive low-pass filter designed for
real-time signal smoothing. It reduces jitter at low speeds while
maintaining responsiveness at high speeds.

Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012). 
    "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
    CHI '12: Proceedings of the SIGCHI Conference on Human Factors in Computing Systems

Mathematical Foundation:
    The filter adapts its cutoff frequency based on signal velocity:
    fc = fc_min + β × |dx/dt|
    
    Where:
    - fc_min: Base cutoff frequency (lower = smoother)
    - β: Speed coefficient (higher = more responsive to fast movements)
    - dx/dt: Rate of change of the signal
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from squat_analyzer.utils.logging import get_logger

logger = get_logger(__name__)


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    """
    Compute the exponential smoothing factor.
    
    Args:
        t_e: Time elapsed since last sample
        cutoff: Cutoff frequency
        
    Returns:
        Smoothing factor alpha in [0, 1]
    """
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


class LowPassFilter:
    """
    Simple first-order low-pass filter.
    
    Uses exponential smoothing to filter a signal.
    """
    
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialize the low-pass filter.
        
        Args:
            alpha: Initial smoothing factor [0, 1]
                   alpha = 1.0 means no filtering
        """
        self._alpha = alpha
        self._initialized = False
        self._value: float = 0.0
    
    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        """
        Filter a value.
        
        Args:
            value: New input value
            alpha: Optional smoothing factor override
            
        Returns:
            Filtered value
        """
        if alpha is not None:
            self._alpha = alpha
        
        if not self._initialized:
            self._value = value
            self._initialized = True
            return value
        
        # Exponential smoothing: y[n] = α×x[n] + (1-α)×y[n-1]
        self._value = self._alpha * value + (1 - self._alpha) * self._value
        return self._value
    
    def reset(self) -> None:
        """Reset filter state."""
        self._initialized = False
        self._value = 0.0
    
    @property
    def value(self) -> float:
        """Current filtered value."""
        return self._value


@dataclass
class OneEuroFilter:
    """
    One Euro Filter for adaptive signal smoothing.
    
    This filter automatically adjusts its smoothing based on signal
    velocity, providing optimal noise reduction while maintaining
    responsiveness.
    
    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz)
                    Lower values = smoother filtering at rest
                    Typical range: 0.5 - 5.0
                    
        beta: Speed coefficient
              Higher values = more responsive to fast movements
              Typical range: 0.0001 - 0.1
              
        d_cutoff: Derivative cutoff frequency (Hz)
                  Controls smoothing of velocity estimation
                  Typical: 1.0
    
    Example:
        >>> filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        >>> for value in noisy_signal:
        ...     smoothed = filter.filter(value)
        ...     print(f"{smoothed:.2f}")
    """
    
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    
    # Internal state
    _x_filter: LowPassFilter = field(default_factory=LowPassFilter, repr=False)
    _dx_filter: LowPassFilter = field(default_factory=LowPassFilter, repr=False)
    _last_time: float = field(default=0.0, repr=False)
    _initialized: bool = field(default=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize internal filters."""
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time = 0.0
        self._initialized = False
    
    def filter(
        self,
        value: float,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Filter a value using the One-Euro algorithm.
        
        Args:
            value: Input value to filter
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            Filtered value
        """
        # Get timestamp
        if timestamp is None:
            timestamp = time.perf_counter()
        
        # First sample initialization
        if not self._initialized:
            self._last_time = timestamp
            self._dx_filter.filter(0.0)
            self._x_filter.filter(value)
            self._initialized = True
            return value
        
        # Calculate time delta
        t_e = timestamp - self._last_time
        if t_e <= 0:
            t_e = 1e-6  # Prevent division by zero
        self._last_time = timestamp
        
        # Estimate velocity (derivative)
        dx = (value - self._x_filter.value) / t_e
        
        # Filter the derivative
        alpha_d = _smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self._dx_filter.filter(dx, alpha_d)
        
        # Compute adaptive cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter the value
        alpha = _smoothing_factor(t_e, cutoff)
        return self._x_filter.filter(value, alpha)
    
    def reset(self) -> None:
        """Reset filter to initial state."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = 0.0
        self._initialized = False
    
    @property
    def value(self) -> float:
        """Current filtered value."""
        return self._x_filter.value


class KeypointFilter:
    """
    Multi-dimensional One-Euro filter for keypoint smoothing.
    
    Applies separate One-Euro filters to each keypoint coordinate,
    enabling smooth tracking of pose landmarks.
    
    Example:
        >>> filter = KeypointFilter(num_keypoints=17)
        >>> for frame_keypoints in detection_stream:
        ...     smooth_kps = filter.filter(frame_keypoints)
    """
    
    def __init__(
        self,
        num_keypoints: int = 17,
        num_dimensions: int = 2,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        """
        Initialize keypoint filter.
        
        Args:
            num_keypoints: Number of keypoints to track
            num_dimensions: Dimensions per keypoint (2 for xy, 3 for xyz)
            min_cutoff: Base cutoff frequency
            beta: Speed coefficient
            d_cutoff: Derivative cutoff
        """
        self.num_keypoints = num_keypoints
        self.num_dimensions = num_dimensions
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # Create filter bank
        self._filters: list[list[OneEuroFilter]] = [
            [
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
                for _ in range(num_dimensions)
            ]
            for _ in range(num_keypoints)
        ]
        
        logger.debug(
            "KeypointFilter initialized",
            num_keypoints=num_keypoints,
            dimensions=num_dimensions,
        )
    
    def filter(
        self,
        keypoints: NDArray[np.float32],
        timestamp: Optional[float] = None,
    ) -> NDArray[np.float32]:
        """
        Filter keypoint array.
        
        Args:
            keypoints: Array of shape (num_keypoints, num_dimensions)
            timestamp: Optional timestamp for all filters
            
        Returns:
            Filtered keypoints array
        """
        if timestamp is None:
            timestamp = time.perf_counter()
        
        keypoints = np.asarray(keypoints, dtype=np.float32)
        result = np.zeros_like(keypoints)
        
        for i in range(min(len(keypoints), self.num_keypoints)):
            for j in range(min(len(keypoints[i]), self.num_dimensions)):
                result[i, j] = self._filters[i][j].filter(
                    float(keypoints[i, j]),
                    timestamp,
                )
        
        return result
    
    def reset(self) -> None:
        """Reset all filters."""
        for kp_filters in self._filters:
            for dim_filter in kp_filters:
                dim_filter.reset()
        logger.debug("KeypointFilter reset")
    
    def update_parameters(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        """
        Update filter parameters on all internal filters.
        
        Args:
            min_cutoff: New minimum cutoff frequency
            beta: New speed coefficient
            d_cutoff: New derivative cutoff
        """
        for kp_filters in self._filters:
            for dim_filter in kp_filters:
                if min_cutoff is not None:
                    dim_filter.min_cutoff = min_cutoff
                if beta is not None:
                    dim_filter.beta = beta
                if d_cutoff is not None:
                    dim_filter.d_cutoff = d_cutoff
