"""
Performance Metrics and Profiling Utilities
============================================

Provides real-time performance monitoring including FPS calculation,
latency tracking, and resource utilization statistics.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass
class PerformanceMetrics:
    """
    Real-time performance metrics tracker.
    
    Tracks FPS, latency, and timing statistics with rolling
    window averages for smooth metric reporting.
    
    Attributes:
        window_size: Number of samples for rolling average
        fps: Current frames per second
        latency_ms: Current processing latency in milliseconds
        avg_fps: Rolling average FPS
        avg_latency_ms: Rolling average latency
        
    Example:
        >>> metrics = PerformanceMetrics()
        >>> metrics.start_frame()
        >>> # ... processing ...
        >>> metrics.end_frame()
        >>> print(f"FPS: {metrics.fps:.1f}")
    """
    
    window_size: int = 30
    
    # Timing storage
    _frame_times: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    _latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    _frame_start: float = field(default=0.0)
    _last_frame_time: float = field(default=0.0)
    
    # Counters
    _total_frames: int = field(default=0)
    _start_time: float = field(default_factory=time.perf_counter)
    
    def __post_init__(self) -> None:
        """Initialize deques with correct maxlen after dataclass creation."""
        self._frame_times = deque(maxlen=self.window_size)
        self._latencies = deque(maxlen=self.window_size)
        self._start_time = time.perf_counter()
    
    def start_frame(self) -> None:
        """Mark the start of frame processing."""
        self._frame_start = time.perf_counter()
    
    def end_frame(self) -> None:
        """Mark the end of frame processing and update metrics."""
        current_time = time.perf_counter()
        
        # Calculate latency (processing time)
        latency = (current_time - self._frame_start) * 1000  # ms
        self._latencies.append(latency)
        
        # Calculate frame time (time between frames)
        if self._last_frame_time > 0:
            frame_time = current_time - self._last_frame_time
            self._frame_times.append(frame_time)
        
        self._last_frame_time = current_time
        self._total_frames += 1
    
    @property
    def fps(self) -> float:
        """Current instantaneous FPS based on recent frame times."""
        if not self._frame_times:
            return 0.0
        return 1.0 / self._frame_times[-1] if self._frame_times[-1] > 0 else 0.0
    
    @property
    def avg_fps(self) -> float:
        """Rolling average FPS."""
        if not self._frame_times:
            return 0.0
        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    @property
    def latency_ms(self) -> float:
        """Current processing latency in milliseconds."""
        return self._latencies[-1] if self._latencies else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Rolling average latency in milliseconds."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)
    
    @property
    def total_frames(self) -> int:
        """Total number of frames processed."""
        return self._total_frames
    
    @property
    def uptime_seconds(self) -> float:
        """Total runtime in seconds."""
        return time.perf_counter() - self._start_time
    
    @property
    def overall_fps(self) -> float:
        """Overall average FPS since start."""
        uptime = self.uptime_seconds
        return self._total_frames / uptime if uptime > 0 else 0.0
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._frame_times.clear()
        self._latencies.clear()
        self._frame_start = 0.0
        self._last_frame_time = 0.0
        self._total_frames = 0
        self._start_time = time.perf_counter()
    
    def get_stats(self) -> dict[str, float]:
        """Get all metrics as a dictionary.
        
        Returns:
            Dictionary containing all current metrics.
        """
        return {
            "fps": self.fps,
            "avg_fps": self.avg_fps,
            "latency_ms": self.latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "total_frames": float(self.total_frames),
            "uptime_seconds": self.uptime_seconds,
            "overall_fps": self.overall_fps,
        }


class ScopedTimer:
    """
    Context manager for timing code blocks.
    
    Example:
        >>> with ScopedTimer("pose_estimation") as timer:
        ...     result = estimate_pose(frame)
        >>> print(f"Took {timer.elapsed_ms:.2f}ms")
    """
    
    def __init__(self, name: str = "operation") -> None:
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
    
    def __enter__(self) -> "ScopedTimer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return self.end_time - self.start_time
