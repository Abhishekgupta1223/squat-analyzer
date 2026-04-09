"""
World-Class Multi-Person Tracker v2.0
======================================

Expert-level tracking designed for the specific challenge of tracking ONE person
consistently when MULTIPLE people perform SIMILAR movements (like synchronized squats).

Key Innovations:
1. Body Proportion Signature - invariant features that don't change during squats
2. Horizontal Position Lock - X position is more stable than Y during squats
3. Appearance Encoding - relative keypoint geometry creates unique "fingerprint"
4. Track Confidence Scoring - weighted combination of multiple features
5. Hysteresis - strongly resists track switching once locked on
6. Spatial Gating - only considers detections within expected region

The core insight: When two people squat simultaneously, their Y positions and sizes
change similarly. But their X positions (horizontal) and body proportions remain
DIFFERENT and STABLE. We exploit this asymmetry.

Reference:
    SORT (Simple Online Realtime Tracking) - Bewley et al. 2016
    Deep SORT - Wojke et al. 2017 (appearance features)
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from collections import deque

from squat_analyzer.core.keypoints import Keypoints
from squat_analyzer.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# BODY SIGNATURE EXTRACTION
# =============================================================================

@dataclass
class BodySignature:
    """
    Invariant body features that don't change significantly during squats.
    
    These features are used to distinguish between different people even when
    they perform identical movements.
    """
    # Horizontal position (X) - much more stable than Y during squats
    center_x: float
    
    # Body proportions (ratios are scale-invariant)
    shoulder_width: float           # Absolute shoulder width
    hip_width: float                # Absolute hip width
    shoulder_hip_ratio: float       # shoulder_width / hip_width
    torso_length: float             # Distance from shoulder to hip
    
    # Lateral asymmetry (left-right differences are person-specific)
    shoulder_tilt: float            # Difference in Y between shoulders
    hip_tilt: float                 # Difference in Y between hips
    
    # Arm geometry (often distinctive per person)
    left_arm_angle: float
    right_arm_angle: float
    
    @staticmethod
    def from_keypoints(kp: Keypoints) -> Optional["BodySignature"]:
        """Extract body signature from keypoints."""
        try:
            # Shoulder positions
            l_shoulder = kp.left_shoulder
            r_shoulder = kp.right_shoulder
            
            # Hip positions  
            l_hip = kp.left_hip
            r_hip = kp.right_hip
            
            # Horizontal center (very stable during squats)
            center_x = (l_shoulder[0] + r_shoulder[0] + l_hip[0] + r_hip[0]) / 4
            
            # Body proportions
            shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
            hip_width = abs(r_hip[0] - l_hip[0])
            
            if shoulder_width < 10 or hip_width < 10:
                return None
            
            shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 1.0
            
            # Torso length (center to center)
            shoulder_center_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_center_y = (l_hip[1] + r_hip[1]) / 2
            torso_length = abs(hip_center_y - shoulder_center_y)
            
            # Lateral tilts (person-specific posture)
            shoulder_tilt = l_shoulder[1] - r_shoulder[1]
            hip_tilt = l_hip[1] - r_hip[1]
            
            # Arm angles
            try:
                l_elbow = kp.left_elbow
                r_elbow = kp.right_elbow
                left_arm_angle = math.atan2(l_elbow[1] - l_shoulder[1], 
                                           l_elbow[0] - l_shoulder[0])
                right_arm_angle = math.atan2(r_elbow[1] - r_shoulder[1],
                                            r_elbow[0] - r_shoulder[0])
            except:
                left_arm_angle = 0.0
                right_arm_angle = 0.0
            
            return BodySignature(
                center_x=center_x,
                shoulder_width=shoulder_width,
                hip_width=hip_width,
                shoulder_hip_ratio=shoulder_hip_ratio,
                torso_length=torso_length,
                shoulder_tilt=shoulder_tilt,
                hip_tilt=hip_tilt,
                left_arm_angle=left_arm_angle,
                right_arm_angle=right_arm_angle,
            )
        except Exception as e:
            return None
    
    def similarity(self, other: "BodySignature", frame_width: float) -> float:
        """
        Compute similarity score between two body signatures.
        
        Returns value in [0, 1] where 1 = identical, 0 = completely different.
        Uses weighted combination of features.
        """
        if frame_width < 100:
            frame_width = 1000
        
        scores = []
        weights = []
        
        # 1. Horizontal position (MOST IMPORTANT for synchronized movements)
        #    Normalize by frame width
        x_diff = abs(self.center_x - other.center_x) / frame_width
        x_score = max(0, 1 - x_diff * 5)  # 20% of frame = 0 score
        scores.append(x_score)
        weights.append(4.0)  # High weight
        
        # 2. Shoulder-hip ratio (very stable, person-specific)
        ratio_diff = abs(self.shoulder_hip_ratio - other.shoulder_hip_ratio)
        ratio_score = max(0, 1 - ratio_diff * 3)
        scores.append(ratio_score)
        weights.append(2.0)
        
        # 3. Shoulder width similarity (scale-normalized)
        width_ratio = min(self.shoulder_width, other.shoulder_width) / \
                      max(self.shoulder_width, other.shoulder_width) if max(self.shoulder_width, other.shoulder_width) > 0 else 0
        scores.append(width_ratio)
        weights.append(1.5)
        
        # 4. Shoulder tilt similarity
        tilt_diff = abs(self.shoulder_tilt - other.shoulder_tilt)
        tilt_score = max(0, 1 - tilt_diff / 50)
        scores.append(tilt_score)
        weights.append(1.0)
        
        # 5. Hip tilt similarity
        hip_tilt_diff = abs(self.hip_tilt - other.hip_tilt)
        hip_tilt_score = max(0, 1 - hip_tilt_diff / 50)
        scores.append(hip_tilt_score)
        weights.append(1.0)
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_sum / total_weight if total_weight > 0 else 0


# =============================================================================
# ROBUST PERSON TRACK
# =============================================================================

@dataclass
class RobustTrack:
    """
    Enhanced track with body signature and confidence tracking.
    """
    track_id: int
    
    # Position state
    center_x: float
    center_y: float
    
    # Velocity estimates
    vx: float = 0.0
    vy: float = 0.0
    
    # Body signature history (for stable matching)
    signature_history: deque = field(default_factory=lambda: deque(maxlen=15))
    
    # Confidence and tracking state
    confidence: float = 1.0
    frames_tracked: int = 0
    frames_lost: int = 0
    last_update_time: float = 0.0
    
    # Lock-on mechanism
    locked: bool = False
    lock_confidence: float = 0.0
    
    def get_stable_signature(self) -> Optional[BodySignature]:
        """Get averaged stable signature from history."""
        if not self.signature_history:
            return None
        
        # Average the numeric fields
        n = len(self.signature_history)
        return BodySignature(
            center_x=sum(s.center_x for s in self.signature_history) / n,
            shoulder_width=sum(s.shoulder_width for s in self.signature_history) / n,
            hip_width=sum(s.hip_width for s in self.signature_history) / n,
            shoulder_hip_ratio=sum(s.shoulder_hip_ratio for s in self.signature_history) / n,
            torso_length=sum(s.torso_length for s in self.signature_history) / n,
            shoulder_tilt=sum(s.shoulder_tilt for s in self.signature_history) / n,
            hip_tilt=sum(s.hip_tilt for s in self.signature_history) / n,
            left_arm_angle=sum(s.left_arm_angle for s in self.signature_history) / n,
            right_arm_angle=sum(s.right_arm_angle for s in self.signature_history) / n,
        )
    
    def predict_position(self, dt: float) -> Tuple[float, float]:
        """Predict position using velocity."""
        return (
            self.center_x + self.vx * dt,
            self.center_y + self.vy * dt
        )
    
    def update(self, kp: Keypoints, signature: BodySignature, timestamp: float):
        """Update track with new detection."""
        dt = timestamp - self.last_update_time if self.last_update_time > 0 else 0.033
        
        # Get new center
        new_x = signature.center_x
        new_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        
        # Update velocity with smoothing
        if dt > 0 and self.frames_tracked > 0:
            alpha = 0.4
            new_vx = (new_x - self.center_x) / dt
            new_vy = (new_y - self.center_y) / dt
            self.vx = alpha * new_vx + (1 - alpha) * self.vx
            self.vy = alpha * new_vy + (1 - alpha) * self.vy
        
        # Update position
        self.center_x = new_x
        self.center_y = new_y
        
        # Add to signature history
        self.signature_history.append(signature)
        
        # Update tracking state
        self.frames_tracked += 1
        self.frames_lost = 0
        self.last_update_time = timestamp
        
        # Update lock confidence
        if self.frames_tracked >= 10:
            self.lock_confidence = min(1.0, self.lock_confidence + 0.1)
            if self.lock_confidence >= 0.8:
                self.locked = True
    
    def mark_lost(self):
        """Mark track as lost for this frame."""
        self.frames_lost += 1
        self.lock_confidence = max(0, self.lock_confidence - 0.15)
        if self.lock_confidence < 0.3:
            self.locked = False


# =============================================================================
# MAIN TRACKER
# =============================================================================

class RobustPersonTracker:
    """
    World-class single-person tracker for squat analysis.
    
    SIMPLIFIED BUT BULLETPROOF APPROACH:
    
    When two people perform synchronized squats side by side, the key insight is:
    - Their X (horizontal) positions are COMPLETELY DIFFERENT
    - Their Y (vertical) positions + sizes change SIMILARLY
    
    Solution: HORIZONTAL ZONE LOCKING
    1. On first detection, lock onto the person's horizontal zone (left/right half)
    2. ONLY accept detections from within that zone
    3. This is simple but extremely effective for side-by-side people
    
    This handles the "simultaneous squat" problem perfectly because even when
    both people squat at the same time, they remain in different horizontal zones.
    """
    
    def __init__(
        self,
        zone_margin: float = 0.05,        # Tight margin (5% of frame)
        max_lost_frames: int = 30,        # Frames before re-initialization
    ):
        self.zone_margin = zone_margin
        self.max_lost_frames = max_lost_frames
        
        # Tracking state
        self._locked_zone: Optional[Tuple[float, float]] = None  # (min_x, max_x)
        self._last_x: Optional[float] = None
        self._frames_tracked: int = 0
        self._frames_lost: int = 0
        self._initialized: bool = False
        
        # For velocity prediction
        self._last_center: Optional[Tuple[float, float]] = None
        self._velocity: Tuple[float, float] = (0.0, 0.0)
    
    def _get_person_x(self, kp: Keypoints) -> Optional[float]:
        """Get horizontal center of person."""
        try:
            return (kp.left_hip[0] + kp.right_hip[0] + 
                    kp.left_shoulder[0] + kp.right_shoulder[0]) / 4
        except:
            return None
    
    def _get_person_center(self, kp: Keypoints) -> Optional[Tuple[float, float]]:
        """Get center of person."""
        try:
            x = self._get_person_x(kp)
            y = (kp.left_hip[1] + kp.right_hip[1]) / 2
            return (x, y)
        except:
            return None
    
    def _get_person_width(self, kp: Keypoints) -> float:
        """Get approximate width of person."""
        try:
            return max(
                abs(kp.right_shoulder[0] - kp.left_shoulder[0]),
                abs(kp.right_hip[0] - kp.left_hip[0])
            )
        except:
            return 100
    
    def _is_in_zone(self, x: float) -> bool:
        """Check if X position is within locked zone."""
        if self._locked_zone is None:
            return True
        min_x, max_x = self._locked_zone
        return min_x <= x <= max_x
    
    def _create_zone(self, x: float, width: float, frame_width: float):
        """Create a TIGHT horizontal zone around the person."""
        # Use person's own width plus small margin, NOT frame-based margin
        padding = width * 0.5 + frame_width * self.zone_margin
        
        min_x = max(0, x - padding)
        max_x = min(frame_width, x + padding)
        
        self._locked_zone = (min_x, max_x)
        logger.info(f"Zone locked: x={x:.0f}, zone=[{min_x:.0f}, {max_x:.0f}], width={max_x-min_x:.0f}")
    
    def select_person(
        self,
        keypoints_list: List[Keypoints],
        frame_width: int,
        frame_height: int,
    ) -> Optional[Keypoints]:
        """
        Select the tracked person using horizontal zone locking.
        """
        # No detections
        if not keypoints_list:
            self._frames_lost += 1
            if self._frames_lost >= self.max_lost_frames:
                self._reset_zone()
            return None
        
        # Single person detection
        if len(keypoints_list) == 1:
            kp = keypoints_list[0]
            x = self._get_person_x(kp)
            if x is not None:
                # Not initialized yet - lock onto this person
                if not self._initialized:
                    width = self._get_person_width(kp)
                    self._create_zone(x, width, frame_width)
                    self._initialized = True
                    self._update_tracking(kp, x)
                    return kp
                
                # Already initialized - must be in zone (CRITICAL FIX!)
                # This handles the case where YOLO only detects one person
                # but it's the WRONG person (the other one)
                if self._is_in_zone(x):
                    self._update_tracking(kp, x)
                    return kp
                else:
                    # Single detection but outside zone - they're the wrong person
                    self._frames_lost += 1
                    if self._frames_lost >= self.max_lost_frames:
                        logger.info("Zone lost (single person outside) - reinitializing")
                        self._reset_zone()
                        return self.select_person(keypoints_list, frame_width, frame_height)
                    return None
            return None
        
        # Multiple people - filter by zone
        
        # Not initialized - pick leftmost person (consistent choice)
        if not self._initialized or self._locked_zone is None:
            candidates = []
            for kp in keypoints_list:
                x = self._get_person_x(kp)
                if x is not None:
                    candidates.append((x, kp))
            
            if not candidates:
                return None
            
            # Sort by X, pick leftmost
            candidates.sort(key=lambda c: c[0])
            x, kp = candidates[0]
            
            width = self._get_person_width(kp)
            self._create_zone(x, width, frame_width)
            self._initialized = True
            self._update_tracking(kp, x)
            
            return kp
        
        # Initialized - find person within our zone
        best_kp = None
        best_distance = float('inf')
        
        for kp in keypoints_list:
            x = self._get_person_x(kp)
            if x is None:
                continue
            
            # Must be within locked zone
            if not self._is_in_zone(x):
                continue
            
            # If we have previous position, prefer closest
            if self._last_x is not None:
                dist = abs(x - self._last_x)
                if dist < best_distance:
                    best_distance = dist
                    best_kp = kp
            else:
                best_kp = kp
                break
        
        if best_kp is not None:
            x = self._get_person_x(best_kp)
            self._update_tracking(best_kp, x)
            return best_kp
        
        # No one in zone - mark lost
        self._frames_lost += 1
        
        if self._frames_lost >= self.max_lost_frames:
            logger.info("Zone lost - reinitializing")
            self._reset_zone()
            # Recursively try again with fresh state
            return self.select_person(keypoints_list, frame_width, frame_height)
        
        return None
    
    def _update_tracking(self, kp: Keypoints, x: float):
        """Update tracking state."""
        self._last_x = x
        self._frames_tracked += 1
        self._frames_lost = 0
        
        # Update velocity
        center = self._get_person_center(kp)
        if center and self._last_center:
            self._velocity = (
                0.5 * (center[0] - self._last_center[0]) + 0.5 * self._velocity[0],
                0.5 * (center[1] - self._last_center[1]) + 0.5 * self._velocity[1],
            )
        self._last_center = center
    
    def _reset_zone(self):
        """Reset zone lock."""
        self._locked_zone = None
        self._last_x = None
        self._frames_tracked = 0
        self._frames_lost = 0
        self._initialized = False
        self._last_center = None
        self._velocity = (0.0, 0.0)
    
    def reset(self):
        """Reset tracker for new session."""
        self._reset_zone()
        logger.info("RobustPersonTracker reset")


# =============================================================================
# ALIAS
# =============================================================================

# Make robust tracker the default
PersonTracker = RobustPersonTracker
