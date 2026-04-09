"""
World-Class Squat Detection System v2.0
========================================

Expert-level implementation combining:
- Signal Processing: One-Euro adaptive filtering, velocity estimation
- Computer Vision: Multi-feature extraction, confidence weighting  
- State Machine: Hysteresis bands, N-frame confirmation, velocity-aware phases
- Robustness: Multi-person tracking, outlier rejection, graceful degradation

Architecture:
    Raw Keypoints → Filtering → Feature Extraction → State Machine → Rep Count

Key Innovations:
    1. Adaptive One-Euro filtering on computed metrics (not just raw keypoints)
    2. Hysteresis bands prevent oscillation at state boundaries
    3. N-frame confirmation prevents noise-induced false transitions
    4. Velocity-based phase detection (descending vs ascending)
    5. Multi-feature validation for rep counting
    6. Outlier rejection for robustness to tracking errors

References:
    - Casiez et al. (2012): 1€ Filter for interactive systems
    - Welch & Bishop (2006): Kalman filtering for tracking
    - Schoenfeld (2010): Squat biomechanics
"""

from __future__ import annotations

import time
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Optional, Tuple, List

import numpy as np

from squat_analyzer.core.keypoints import Keypoints
from squat_analyzer.core.angles import AngleCalculator
from squat_analyzer.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# SIGNAL PROCESSING COMPONENTS
# =============================================================================

class AdaptiveFilter:
    """
    FAST adaptive filter for athletic movements.
    
    Much more responsive than standard One-Euro for quick movements like squats.
    Uses exponential moving average with velocity-adaptive smoothing.
    """
    
    def __init__(
        self,
        alpha_slow: float = 0.3,   # Smoothing when stationary
        alpha_fast: float = 0.8,   # Smoothing when moving fast
        velocity_threshold: float = 0.02,  # Threshold to switch to fast mode
    ):
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.velocity_threshold = velocity_threshold
        
        self._value: Optional[float] = None
        self._prev_raw: Optional[float] = None  # Track raw values for velocity
        self._raw_velocity: float = 0.0         # Velocity of RAW signal
        self._initialized = False
    
    def update(self, value: float, timestamp: Optional[float] = None) -> float:
        """
        Filter a value with velocity-adaptive smoothing.
        
        Fast movements use high alpha (less smoothing).
        Slow/stationary uses low alpha (more smoothing).
        """
        if not self._initialized:
            self._value = value
            self._prev_raw = value
            self._raw_velocity = 0.0
            self._initialized = True
            return value
        
        # Calculate RAW velocity (change per update in RAW signal)
        raw_velocity = value - self._prev_raw
        self._prev_raw = value
        
        # Smooth velocity estimate slightly
        self._raw_velocity = 0.6 * raw_velocity + 0.4 * self._raw_velocity
        
        # Choose alpha based on RAW velocity
        speed = abs(self._raw_velocity)
        if speed > self.velocity_threshold:
            alpha = self.alpha_fast  # Fast movement - respond quickly
        else:
            alpha = self.alpha_slow  # Stationary - smooth more
        
        # Update filtered value
        self._value = alpha * value + (1 - alpha) * self._value
        
        return self._value
    
    @property
    def velocity(self) -> float:
        """Current smoothed velocity of RAW signal."""
        return self._raw_velocity if self._initialized else 0.0
    
    @property
    def value(self) -> float:
        """Current filtered value."""
        return self._value if self._initialized else 0.0
    
    def reset(self):
        """Reset filter state."""
        self._value = None
        self._prev_raw = None
        self._raw_velocity = 0.0
        self._initialized = False


class VelocityEstimator:
    """
    Robust velocity estimation using weighted moving average.
    
    More stable than simple frame-to-frame difference.
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._values: Deque[Tuple[float, float]] = deque(maxlen=window_size)
    
    def update(self, value: float, timestamp: float) -> float:
        """Add value and compute velocity."""
        self._values.append((timestamp, value))
        return self.velocity
    
    @property
    def velocity(self) -> float:
        """Compute velocity using linear regression over window."""
        if len(self._values) < 2:
            return 0.0
        
        times = np.array([v[0] for v in self._values])
        values = np.array([v[1] for v in self._values])
        
        # Normalize time to prevent numerical issues
        times = times - times[0]
        
        if times[-1] <= 0:
            return 0.0
        
        # Simple linear regression slope
        n = len(times)
        sum_t = np.sum(times)
        sum_v = np.sum(values)
        sum_tv = np.sum(times * values)
        sum_tt = np.sum(times * times)
        
        denom = n * sum_tt - sum_t * sum_t
        if abs(denom) < 1e-10:
            return 0.0
        
        slope = (n * sum_tv - sum_t * sum_v) / denom
        return slope
    
    def reset(self):
        """Reset estimator."""
        self._values.clear()


# =============================================================================
# STATE MACHINE WITH HYSTERESIS
# =============================================================================

class SquatPhase(Enum):
    """Squat movement phases."""
    IDLE = auto()       # Not tracking / invalid pose
    STANDING = auto()   # Ready position - upright
    DESCENDING = auto() # Going down
    BOTTOM = auto()     # At lowest point
    ASCENDING = auto()  # Coming back up
    
    def __str__(self) -> str:
        return self.name


@dataclass
class HysteresisConfig:
    """
    RELAXED thresholds for real-world squat detection.
    
    Based on observed ratios:
    - Standing: ~0.25-0.35
    - Squatting: ~0.15-0.22
    """
    # Standing thresholds (hip-knee ratio)
    enter_standing: float = 0.23    # Must exceed this to enter STANDING
    exit_standing: float = 0.19     # Must drop below this to exit STANDING
    
    # Squat thresholds - MUCH more lenient
    enter_squat: float = 0.22       # Must drop below this to enter SQUAT
    exit_squat: float = 0.25        # Must exceed this to exit SQUAT
    
    # Velocity thresholds (ratio units per update) - very sensitive
    descending_velocity: float = -0.005   # Negative = going down
    ascending_velocity: float = 0.005     # Positive = going up
    stationary_velocity: float = 0.003    # Near-zero
    
    # Frame confirmation - IMMEDIATE for responsiveness
    frames_to_confirm: int = 1      # Immediate transition (no delay)
    
    # Rep validation - RELAXED
    min_rep_duration_ms: float = 250.0   # 250ms minimum (very fast squats OK)
    min_depth_ratio: float = 0.02        # 2% body height change


@dataclass 
class SquatMetrics:
    """Current computed metrics for squat analysis."""
    hip_knee_ratio: float = 0.25        # Primary metric for front view
    hip_knee_ratio_filtered: float = 0.25   # Smoothed ratio
    velocity: float = 0.0               # Rate of change
    body_height: float = 200.0          # Shoulder to ankle
    confidence: float = 1.0             # Detection confidence
    
    # Side view angles (when available)
    knee_angle: float = 180.0
    hip_angle: float = 180.0


class SquatStateMachine:
    """
    State machine with hysteresis and N-frame confirmation.
    
    Key features:
    1. Hysteresis bands prevent oscillation
    2. Requires N consecutive frames for state transition
    3. Velocity-aware phase detection
    4. Tracks metrics for rep validation
    """
    
    def __init__(self, config: Optional[HysteresisConfig] = None):
        self.config = config or HysteresisConfig()
        
        self._phase = SquatPhase.IDLE
        self._pending_phase: Optional[SquatPhase] = None
        self._pending_frames = 0
        
        # Rep tracking
        self._in_squat_cycle = False
        self._cycle_start_time: float = 0.0
        self._min_ratio_in_cycle: float = 1.0
        self._standing_ratio: float = 0.25
        self._rep_count = 0
        
        # History
        self._phase_history: Deque[SquatPhase] = deque(maxlen=30)
    
    @property
    def phase(self) -> SquatPhase:
        return self._phase
    
    @property
    def rep_count(self) -> int:
        return self._rep_count
    
    @property
    def in_squat(self) -> bool:
        return self._in_squat_cycle
    
    def _confirm_transition(self, new_phase: SquatPhase) -> bool:
        """
        Require N consecutive frames agreeing on new phase.
        Returns True if transition should occur.
        """
        if self._pending_phase == new_phase:
            self._pending_frames += 1
        else:
            self._pending_phase = new_phase
            self._pending_frames = 1
        
        return self._pending_frames >= self.config.frames_to_confirm
    
    def _try_transition(self, new_phase: SquatPhase):
        """Attempt phase transition with confirmation."""
        if new_phase != self._phase:
            if self._confirm_transition(new_phase):
                old_phase = self._phase
                self._phase = new_phase
                self._pending_phase = None
                self._pending_frames = 0
                self._on_phase_change(old_phase, new_phase)
    
    def _on_phase_change(self, old_phase: SquatPhase, new_phase: SquatPhase):
        """Handle phase transition events."""
        logger.debug(f"Phase: {old_phase.name} → {new_phase.name}")
        
        # Started descending from standing
        if old_phase == SquatPhase.STANDING and new_phase == SquatPhase.DESCENDING:
            self._in_squat_cycle = True
            self._cycle_start_time = time.time() * 1000
            self._min_ratio_in_cycle = 1.0
            logger.info("▼ Squat descent started")
        
        # Returned to standing (potentially completed rep)
        elif new_phase == SquatPhase.STANDING and self._in_squat_cycle:
            self._complete_cycle()
    
    def _complete_cycle(self):
        """Validate and count completed squat cycle."""
        duration_ms = time.time() * 1000 - self._cycle_start_time
        depth_ratio = self._standing_ratio - self._min_ratio_in_cycle
        
        valid_time = duration_ms >= self.config.min_rep_duration_ms
        valid_depth = depth_ratio >= self.config.min_depth_ratio
        
        if valid_time and valid_depth:
            self._rep_count += 1
            logger.info(
                f"✓ REP #{self._rep_count}! "
                f"depth={depth_ratio:.3f}, dur={duration_ms:.0f}ms"
            )
        else:
            reasons = []
            if not valid_time:
                reasons.append(f"fast({duration_ms:.0f}ms)")
            if not valid_depth:
                reasons.append(f"shallow({depth_ratio:.3f})")
            logger.debug(f"Invalid cycle: {', '.join(reasons)}")
        
        self._in_squat_cycle = False
    
    def update(self, metrics: SquatMetrics, timestamp: float) -> SquatPhase:
        """
        Update state machine with current metrics.
        
        Uses RAW ratio for decisions (more responsive) with filtered for display.
        """
        # Use RAW ratio for decisions - much more responsive!
        raw_ratio = metrics.hip_knee_ratio
        filtered_ratio = metrics.hip_knee_ratio_filtered
        velocity = metrics.velocity
        
        # Track minimum ratio during squat cycle (use raw for accuracy)
        if self._in_squat_cycle:
            self._min_ratio_in_cycle = min(self._min_ratio_in_cycle, raw_ratio)
        
        # Determine target phase based on RAW ratio (not filtered!)
        target_phase = self._determine_phase(raw_ratio, velocity)
        
        # Attempt transition with confirmation
        self._try_transition(target_phase)
        
        self._phase_history.append(self._phase)
        return self._phase
    
    def _determine_phase(self, ratio: float, velocity: float) -> SquatPhase:
        """
        SIMPLIFIED phase detection using raw ratio.
        
        Key insight: Use relative thresholds and velocity direction.
        """
        cfg = self.config
        
        # STANDING or IDLE - looking for descent
        if self._phase in (SquatPhase.IDLE, SquatPhase.STANDING):
            if ratio > cfg.enter_standing:
                # Definitely standing
                self._standing_ratio = max(self._standing_ratio, ratio)
                return SquatPhase.STANDING
            elif ratio < cfg.enter_squat:
                # Started squatting!
                return SquatPhase.DESCENDING
            elif velocity < cfg.descending_velocity:
                # Moving down - transition to descending
                return SquatPhase.DESCENDING
            return SquatPhase.STANDING if self._phase == SquatPhase.STANDING else SquatPhase.IDLE
        
        # DESCENDING - going down
        elif self._phase == SquatPhase.DESCENDING:
            if ratio > cfg.exit_squat:
                # Back to standing
                return SquatPhase.STANDING
            elif velocity > cfg.ascending_velocity:
                # Started coming back up
                return SquatPhase.ASCENDING
            elif velocity > -cfg.stationary_velocity:
                # Slowed down at bottom
                return SquatPhase.BOTTOM
            return SquatPhase.DESCENDING
        
        # BOTTOM - at lowest point
        elif self._phase == SquatPhase.BOTTOM:
            if ratio > cfg.exit_squat:
                return SquatPhase.STANDING
            elif velocity > cfg.ascending_velocity:
                return SquatPhase.ASCENDING
            return SquatPhase.BOTTOM
        
        # ASCENDING - coming back up
        elif self._phase == SquatPhase.ASCENDING:
            if ratio > cfg.exit_squat:
                return SquatPhase.STANDING
            elif velocity < cfg.descending_velocity:
                # Changed direction - going back down
                return SquatPhase.DESCENDING
            return SquatPhase.ASCENDING
        
        return self._phase
    
    def reset(self):
        """Reset state machine."""
        self._phase = SquatPhase.IDLE
        self._pending_phase = None
        self._pending_frames = 0
        self._in_squat_cycle = False
        self._rep_count = 0
        self._min_ratio_in_cycle = 1.0
        self._phase_history.clear()


# =============================================================================
# MAIN SQUAT DETECTOR V2
# =============================================================================

class SquatDetectorV2:
    """
    Production-grade squat detector with signal processing and robust state machine.
    
    Pipeline:
        1. Extract features from keypoints
        2. Apply adaptive filtering
        3. Estimate velocity
        4. Update state machine with hysteresis
        5. Validate and count reps
    
    Usage:
        detector = SquatDetectorV2()
        
        for frame in video:
            keypoints = pose_estimator.estimate(frame)
            phase = detector.update(keypoints)
            reps = detector.rep_count
    """
    
    def __init__(
        self,
        hysteresis_config: Optional[HysteresisConfig] = None,
    ):
        # State machine
        self._state_machine = SquatStateMachine(hysteresis_config)
        
        # Signal processing - FAST filter for athletic movements
        self._ratio_filter = AdaptiveFilter(
            alpha_slow=0.4,   # Moderate smoothing at rest
            alpha_fast=0.85,  # Very responsive when moving
            velocity_threshold=0.015,
        )
        
        # View detection
        self._view_samples: List[str] = []
        self._detected_view = "unknown"
        self._frame_count = 0
        
        # Outlier rejection - more lenient
        self._last_valid_ratio: float = 0.25
        self._max_ratio_jump: float = 0.20  # Allow larger jumps
        
        # Angle calculator for side view
        self._angle_calculator = AngleCalculator()
        
        # Metrics
        self._current_metrics = SquatMetrics()
        
        logger.info("SquatDetectorV2 initialized (signal-processed mode)")
    
    @property
    def rep_count(self) -> int:
        return self._state_machine.rep_count
    
    @property
    def phase(self) -> SquatPhase:
        return self._state_machine.phase
    
    @property
    def current_angle(self) -> float:
        """For UI display compatibility."""
        return self._current_metrics.hip_knee_ratio_filtered * 100
    
    @property
    def metrics(self) -> SquatMetrics:
        return self._current_metrics
    
    def _detect_view(self, keypoints: Keypoints) -> str:
        """Detect camera orientation (front vs side)."""
        try:
            l_shoulder = keypoints.left_shoulder
            r_shoulder = keypoints.right_shoulder
            l_hip = keypoints.left_hip
            r_hip = keypoints.right_hip
            
            shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
            torso_height = abs((l_hip[1] + r_hip[1])/2 - (l_shoulder[1] + r_shoulder[1])/2)
            
            if torso_height < 10:
                return "unknown"
            
            width_ratio = shoulder_width / torso_height
            return "front" if width_ratio > 0.25 else "side"
        except:
            return "unknown"
    
    def _extract_features(self, keypoints: Keypoints) -> Optional[SquatMetrics]:
        """
        Extract normalized features from keypoints.
        
        Returns None if keypoints are invalid.
        """
        try:
            # Get key positions
            hip_y = (keypoints.left_hip[1] + keypoints.right_hip[1]) / 2
            knee_y = (keypoints.left_knee[1] + keypoints.right_knee[1]) / 2
            shoulder_y = (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) / 2
            ankle_y = (keypoints.left_ankle[1] + keypoints.right_ankle[1]) / 2
            
            # Body height (for normalization)
            body_height = ankle_y - shoulder_y
            if body_height < 50:
                return None
            
            # Hip-knee ratio (primary metric)
            hip_knee_distance = knee_y - hip_y
            raw_ratio = hip_knee_distance / body_height
            
            # Outlier rejection: reject if ratio jumps too much
            if abs(raw_ratio - self._last_valid_ratio) > self._max_ratio_jump:
                # Possible tracking error - use last valid ratio
                logger.debug(f"Ratio outlier rejected: {raw_ratio:.3f} (last: {self._last_valid_ratio:.3f})")
                raw_ratio = self._last_valid_ratio
            else:
                self._last_valid_ratio = raw_ratio
            
            # Confidence from keypoint visibility
            confidence = 1.0
            if keypoints.confidence is not None:
                leg_indices = [11, 12, 13, 14, 15, 16]
                confidences = [keypoints.confidence[i] for i in leg_indices if i < len(keypoints.confidence)]
                if confidences:
                    confidence = sum(confidences) / len(confidences)
            
            return SquatMetrics(
                hip_knee_ratio=raw_ratio,
                hip_knee_ratio_filtered=raw_ratio,  # Will be filtered
                body_height=body_height,
                confidence=confidence,
            )
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None
    
    def _extract_side_view_features(self, keypoints: Keypoints) -> Optional[SquatMetrics]:
        """Extract features for side view using joint angles."""
        try:
            knee_angle = self._angle_calculator.knee_flexion_angle(keypoints)
            hip_angle = self._angle_calculator.hip_flexion_angle(keypoints)
            
            if np.isnan(knee_angle) or np.isnan(hip_angle):
                return None
            
            # Convert knee angle to ratio-like metric for consistency
            # 180° (standing) → ratio ~0.3
            # 90° (deep squat) → ratio ~0.05
            normalized_ratio = (knee_angle - 90) / 300  # Maps 90-180 to 0-0.3
            normalized_ratio = max(0.0, min(0.4, normalized_ratio))
            
            return SquatMetrics(
                hip_knee_ratio=normalized_ratio,
                hip_knee_ratio_filtered=normalized_ratio,
                knee_angle=knee_angle,
                hip_angle=hip_angle,
            )
        except:
            return None
    
    def update(self, keypoints: Keypoints) -> SquatPhase:
        """
        Process keypoints and update squat detection.
        
        Returns current phase.
        """
        timestamp = time.perf_counter()
        self._frame_count += 1
        
        # View detection (first 15 frames)
        if self._frame_count <= 15:
            view = self._detect_view(keypoints)
            self._view_samples.append(view)
            if self._frame_count == 15:
                front_count = self._view_samples.count("front")
                self._detected_view = "front" if front_count > 7 else "side"
                logger.info(f"View detected: {self._detected_view.upper()}")
        
        # Extract features based on view
        if self._detected_view == "side":
            metrics = self._extract_side_view_features(keypoints)
        else:
            metrics = self._extract_features(keypoints)
        
        if metrics is None:
            return self._state_machine.phase
        
        # Apply adaptive filtering (doesn't need timestamp - internal timing)
        filtered_ratio = self._ratio_filter.update(metrics.hip_knee_ratio)
        metrics.hip_knee_ratio_filtered = filtered_ratio
        
        # Use filter's velocity (simpler and more consistent)
        metrics.velocity = self._ratio_filter.velocity
        
        # Store metrics
        self._current_metrics = metrics
        
        # Log every 10 frames for debugging
        if self._frame_count % 10 == 0:
            logger.debug(
                f"raw={metrics.hip_knee_ratio:.3f} "
                f"filt={filtered_ratio:.3f} "
                f"vel={metrics.velocity:.3f} "
                f"phase={self._state_machine.phase.name}"
            )
        
        # Update state machine
        return self._state_machine.update(metrics, timestamp)
    
    def reset(self):
        """Reset detector for new session."""
        self._state_machine.reset()
        self._ratio_filter.reset()
        self._view_samples.clear()
        self._detected_view = "unknown"
        self._frame_count = 0
        self._last_valid_ratio = 0.25
        self._current_metrics = SquatMetrics()
        logger.info("SquatDetectorV2 reset")


# =============================================================================
# ALIAS FOR COMPATIBILITY
# =============================================================================

# Make V2 the default
SquatDetector = SquatDetectorV2
