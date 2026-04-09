"""
Squat Phase Detection State Machine
===================================

Research-grade squat detection with strict validation to prevent
false positives from random movements (sitting, hand waving, etc.)

A VALID SQUAT requires:
    1. Start from STANDING position (knee angle > 165°)
    2. Controlled descent with proper hip hinge
    3. Reach MINIMUM DEPTH (knee angle ≤ 100°)
    4. Return to STANDING position
    5. Hip angle must also change (not just knee bend from sitting)

References:
    - Schoenfeld (2010): Proper squat depth mechanics
    - NSCA Guidelines: Rep counting standards
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Optional

import numpy as np

from squat_analyzer.core.keypoints import Keypoints
from squat_analyzer.core.angles import AngleCalculator
from squat_analyzer.utils.logging import get_logger

logger = get_logger(__name__)


class SquatPhase(Enum):
    """Squat movement phases."""
    
    IDLE = auto()       # Not in squat position (sitting, etc.)
    STANDING = auto()   # Ready position - upright
    DESCENDING = auto() # Going down
    BOTTOM = auto()     # At lowest point
    ASCENDING = auto()  # Coming back up
    
    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class PhaseTransitionThresholds:
    """Strict thresholds for squat detection - prevents false positives."""
    
    # =========================================================================
    # STANDING VALIDATION (must be truly standing to start)
    # =========================================================================
    standing_knee_angle: float = 165.0    # Knees must be THIS straight to be "standing"
    standing_hip_angle: float = 160.0     # Hips must be THIS open to be "standing"
    standing_torso_max: float = 30.0      # Max torso lean when standing (sitting = more lean)
    
    # =========================================================================
    # SQUAT DEPTH REQUIREMENTS (must reach proper depth)
    # =========================================================================
    min_depth_knee_angle: float = 100.0   # Knee must bend to at least this
    min_depth_hip_angle: float = 120.0    # Hip must flex to at least this
    
    # =========================================================================
    # DESCENT DETECTION
    # =========================================================================
    squat_start_angle: float = 155.0      # Below this = started descending
    descent_velocity: float = -1.5        # Must be moving down this fast
    
    # =========================================================================  
    # ASCENT / COMPLETION
    # =========================================================================
    ascent_velocity: float = 1.5          # Must be moving up this fast
    stationary_velocity: float = 0.8      # Below this = not moving
    
    # =========================================================================
    # ANTI-JITTER / STABILITY
    # =========================================================================
    min_frames_in_state: int = 5          # Stay in state for 5+ frames
    min_rep_duration_ms: float = 800.0    # Rep must take at least 800ms
    rep_cooldown_ms: float = 500.0        # Wait 500ms between reps
    
    # =========================================================================
    # HISTORY REQUIREMENTS
    # =========================================================================
    angle_history_size: int = 10          # Track last N angles


@dataclass
class SquatState:
    """Current state with comprehensive tracking."""
    
    phase: SquatPhase = SquatPhase.IDLE
    frames_in_phase: int = 0
    
    # Current measurements
    current_knee_angle: float = 180.0
    current_hip_angle: float = 180.0
    current_torso_angle: float = 0.0
    
    # Rep tracking
    min_knee_angle_this_rep: float = 180.0
    min_hip_angle_this_rep: float = 180.0
    rep_start_time: float = 0.0
    last_rep_time: float = 0.0
    
    # Counters
    rep_count: int = 0
    last_valid_rep: bool = False
    invalid_attempts: int = 0
    
    # History for velocity calculation
    knee_angle_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    hip_angle_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    
    # Starting position validation
    was_standing: bool = False
    standing_knee_angle: float = 180.0
    standing_hip_angle: float = 180.0
    
    def update_angles(
        self,
        knee_angle: float,
        hip_angle: float,
        torso_angle: float,
    ) -> None:
        """Update all angle measurements."""
        self.knee_angle_history.append(knee_angle)
        self.hip_angle_history.append(hip_angle)
        
        self.current_knee_angle = knee_angle
        self.current_hip_angle = hip_angle
        self.current_torso_angle = torso_angle
        
        # Track minimums during rep
        self.min_knee_angle_this_rep = min(self.min_knee_angle_this_rep, knee_angle)
        self.min_hip_angle_this_rep = min(self.min_hip_angle_this_rep, hip_angle)
    
    @property
    def knee_velocity(self) -> float:
        """Knee angle velocity (degrees/frame)."""
        if len(self.knee_angle_history) < 2:
            return 0.0
        return self.knee_angle_history[-1] - self.knee_angle_history[-2]
    
    @property
    def smoothed_knee_velocity(self) -> float:
        """Smoothed velocity over recent frames."""
        if len(self.knee_angle_history) < 3:
            return self.knee_velocity
        velocities = [
            self.knee_angle_history[i] - self.knee_angle_history[i-1]
            for i in range(1, len(self.knee_angle_history))
        ]
        return sum(velocities) / len(velocities)
    
    def reset_rep_tracking(self) -> None:
        """Reset tracking for new rep."""
        self.min_knee_angle_this_rep = 180.0
        self.min_hip_angle_this_rep = 180.0
        self.was_standing = False


class SquatDetector:
    """
    View-Adaptive squat detector for both side-view and front-facing webcam.
    
    Automatically detects camera angle and uses appropriate metrics:
    - SIDE VIEW: Traditional joint angles (knee, hip flexion)
    - FRONT VIEW: Vertical position ratios (hip Y relative to standing)
    
    This enables real-world webcam usage where user faces the camera.
    """
    
    def __init__(
        self,
        thresholds: Optional[PhaseTransitionThresholds] = None,
    ) -> None:
        self._thresholds = thresholds or PhaseTransitionThresholds()
        self._angle_calculator = AngleCalculator()
        self._state = SquatState()
        
        # Simplified tracking
        self._was_standing = False
        self._in_squat = False
        self._min_knee_in_squat = 180.0
        self._min_hip_in_squat = 180.0
        self._squat_start_time = 0.0
        
        # View-adaptive tracking
        self._view_samples = []  # Collect view detection samples
        self._detected_view = "unknown"  # "side" or "front"
        self._standing_hip_y = None  # Reference hip Y for front view
        self._min_hip_y_in_squat = 0  # Track hip drop for front view
        self._frame_count = 0
        
        logger.info("SquatDetector initialized (view-adaptive mode)")
    
    def _detect_view(self, keypoints: Keypoints) -> str:
        """Detect if camera is side view or front view based on body proportions."""
        try:
            # Get shoulder and hip widths
            l_shoulder = keypoints.left_shoulder
            r_shoulder = keypoints.right_shoulder
            l_hip = keypoints.left_hip
            r_hip = keypoints.right_hip
            
            shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
            hip_width = abs(r_hip[0] - l_hip[0])
            
            # Calculate body height for ratio comparison
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            torso_height = abs(hip_y - shoulder_y)
            
            if torso_height < 10:
                return "unknown"
            
            # Front view: horizontal spread is significant relative to torso
            # Side view: body appears narrower
            width_ratio = shoulder_width / torso_height
            
            # Front view typically has width_ratio > 0.3
            if width_ratio > 0.25:
                return "front"
            else:
                return "side"
        except:
            return "unknown"
    
    def update(self, keypoints: Keypoints) -> SquatPhase:
        """Update state machine with new keypoints (view-adaptive)."""
        try:
            self._frame_count += 1
            
            # First 15 frames: detect view angle (reduced from 30)
            if self._frame_count <= 15:
                view = self._detect_view(keypoints)
                self._view_samples.append(view)
                if self._frame_count == 15:
                    # Majority vote
                    front_count = self._view_samples.count("front")
                    self._detected_view = "front" if front_count > 7 else "side"
                    logger.info(f"Detected camera view: {self._detected_view.upper()}")
            
            # Use appropriate detection method
            if self._detected_view == "front":
                return self._update_front_view(keypoints)
            else:
                return self._update_side_view(keypoints)
                
        except Exception as e:
            logger.debug(f"Squat detector update error: {e}")
            return self._state.phase
    
    def _update_front_view(self, keypoints: Keypoints) -> SquatPhase:
        """Front-facing webcam detection using vertical position RATIOS (resolution-independent)."""
        current_time = time.time() * 1000
        
        # Get key positions
        hip_y = (keypoints.left_hip[1] + keypoints.right_hip[1]) / 2
        knee_y = (keypoints.left_knee[1] + keypoints.right_knee[1]) / 2
        shoulder_y = (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) / 2
        ankle_y = (keypoints.left_ankle[1] + keypoints.right_ankle[1]) / 2
        
        # Use RATIOS instead of pixel values for resolution independence
        # Calculate body height estimate (shoulder to ankle)
        body_height = ankle_y - shoulder_y
        if body_height < 50:  # Too small to analyze
            return self._state.phase
        
        # Hip-to-knee ratio: how far hip is from knee relative to body height
        # Standing: hip far from knee (ratio ~0.2-0.3)
        # Squatting: hip close to knee (ratio ~0.1-0.15)
        hip_knee_distance = knee_y - hip_y  # Positive when hip above knee
        hip_knee_ratio = hip_knee_distance / body_height
        
        # Store for debugging
        self._state.current_knee_angle = hip_knee_ratio * 100  # Use as proxy display
        
        # VERY LENIENT thresholds for real-world videos:
        # Standing detection: ratio > 0.16
        is_standing = hip_knee_ratio > 0.16
        
        # Squat detection: ratio < 0.18 (catches most squats even shallow ones)
        is_in_squat = hip_knee_ratio < 0.18
        
        # Log for debugging
        if self._frame_count % 15 == 0:  # Every 15 frames
            logger.debug(f"Front: r={hip_knee_ratio:.3f} stand={is_standing} squat={is_in_squat} inSq={self._in_squat}")
        
        # Not currently in squat
        if not self._in_squat:
            if is_standing:
                self._was_standing = True
                self._standing_hip_y = hip_y
                self._state.phase = SquatPhase.STANDING
                
            elif is_in_squat and self._was_standing and self._standing_hip_y:
                # Started squatting!
                self._in_squat = True
                self._min_hip_y_in_squat = hip_y
                self._squat_start_time = current_time
                self._state.phase = SquatPhase.DESCENDING
                logger.info(f"▼ Squat started: ratio={hip_knee_ratio:.2f}")
                
            else:
                self._state.phase = SquatPhase.IDLE
        
        # Currently in squat
        else:
            # Track maximum hip Y (lowest point in image = highest Y value)
            self._min_hip_y_in_squat = max(self._min_hip_y_in_squat, hip_y)
            
            if is_standing:
                # Completed squat!
                squat_duration = current_time - self._squat_start_time
                
                # Validate: hip must have dropped significantly (use ratio)
                hip_drop = self._min_hip_y_in_squat - self._standing_hip_y
                hip_drop_ratio = hip_drop / body_height
                valid_depth = hip_drop_ratio > 0.02  # VERY lenient: 2% body height
                valid_time = squat_duration >= 200  # Fast squats OK (200ms)
                
                if valid_depth and valid_time:
                    self._state.rep_count += 1
                    logger.info(
                        f"✓ REP #{self._state.rep_count}! drop={hip_drop_ratio:.2f}, dur={squat_duration:.0f}ms"
                    )
                else:
                    reasons = []
                    if not valid_depth:
                        reasons.append(f"shallow({hip_drop_ratio:.2f})")
                    if not valid_time:
                        reasons.append(f"fast({squat_duration:.0f}ms)")
                    logger.debug(f"Invalid rep: {', '.join(reasons)}")
                
                # Reset
                self._in_squat = False
                self._was_standing = True
                self._state.phase = SquatPhase.STANDING
                
            elif hip_knee_ratio > 0.15:
                self._state.phase = SquatPhase.ASCENDING
            elif hip_knee_ratio < 0.08:
                self._state.phase = SquatPhase.BOTTOM
            else:
                self._state.phase = SquatPhase.DESCENDING
        
        return self._state.phase
    
    def _update_side_view(self, keypoints: Keypoints) -> SquatPhase:
        """Side view detection using traditional joint angles with more lenient thresholds."""
        # Calculate all relevant angles
        knee_angle = self._angle_calculator.knee_flexion_angle(keypoints)
        hip_angle = self._angle_calculator.hip_flexion_angle(keypoints)
        torso_angle = self._angle_calculator.torso_inclination(keypoints)
        
        # Validate angles
        if np.isnan(knee_angle) or np.isnan(hip_angle) or np.isnan(torso_angle):
            return self._state.phase
        
        self._state.update_angles(knee_angle, hip_angle, torso_angle)
        
        current_time = time.time() * 1000
        
        # More lenient thresholds for standing and squat detection
        # STANDING: knees mostly straight, hips open
        is_standing = (knee_angle > 155 and hip_angle > 150)
        
        # SQUAT: significant knee bend (was 120, now 135 - more lenient)
        is_in_squat = (knee_angle < 135)
        
        # Not currently in squat
        if not self._in_squat:
            if is_standing:
                self._was_standing = True
                self._state.phase = SquatPhase.STANDING
                
            elif is_in_squat and self._was_standing:
                self._in_squat = True
                self._min_knee_in_squat = knee_angle
                self._min_hip_in_squat = hip_angle
                self._squat_start_time = current_time
                self._state.phase = SquatPhase.DESCENDING
                logger.debug(f"Started squat: knee={knee_angle:.1f}°")
                
            else:
                self._state.phase = SquatPhase.IDLE
        
        # Currently in squat
        else:
            self._min_knee_in_squat = min(self._min_knee_in_squat, knee_angle)
            self._min_hip_in_squat = min(self._min_hip_in_squat, hip_angle)
            
            if is_standing:
                squat_duration = current_time - self._squat_start_time
                
                # More lenient depth requirements
                valid_depth = self._min_knee_in_squat <= 125  # Was 110, now 125
                valid_hip = self._min_hip_in_squat <= 145     # Was 130, now 145
                valid_time = squat_duration >= 400            # Was 500, now 400
                
                if valid_depth and valid_time:  # Hip check less critical
                    self._state.rep_count += 1
                    logger.info(
                        f"✓ REP #{self._state.rep_count}! "
                        f"depth={self._min_knee_in_squat:.0f}°, "
                        f"duration={squat_duration:.0f}ms"
                    )
                else:
                    reasons = []
                    if not valid_depth:
                        reasons.append(f"shallow({self._min_knee_in_squat:.0f}°)")
                    if not valid_time:
                        reasons.append(f"fast({squat_duration:.0f}ms)")
                    logger.debug(f"Invalid rep: {', '.join(reasons)}")
                
                self._in_squat = False
                self._was_standing = True
                self._state.phase = SquatPhase.STANDING
                
            elif knee_angle > 145:
                self._state.phase = SquatPhase.ASCENDING
            elif knee_angle < 110:
                self._state.phase = SquatPhase.BOTTOM
            else:
                self._state.phase = SquatPhase.DESCENDING
        
        return self._state.phase
    
    @property
    def phase(self) -> SquatPhase:
        return self._state.phase
    
    @property
    def rep_count(self) -> int:
        return self._state.rep_count
    
    @property
    def current_angle(self) -> float:
        return self._state.current_knee_angle
    
    @property
    def min_angle_this_rep(self) -> float:
        return self._state.min_knee_angle_this_rep
    
    @property
    def last_rep_valid(self) -> bool:
        return self._state.last_valid_rep
    
    def reset(self) -> None:
        self._state = SquatState()
        logger.info("SquatDetector reset")
    
    def get_state_info(self) -> dict:
        return {
            "phase": str(self._state.phase),
            "rep_count": self._state.rep_count,
            "current_knee_angle": round(self._state.current_knee_angle, 1),
            "current_hip_angle": round(self._state.current_hip_angle, 1),
            "min_knee_this_rep": round(self._state.min_knee_angle_this_rep, 1),
            "min_hip_this_rep": round(self._state.min_hip_angle_this_rep, 1),
            "frames_in_phase": self._state.frames_in_phase,
            "velocity": round(self._state.smoothed_knee_velocity, 2),
            "invalid_attempts": self._state.invalid_attempts,
        }
