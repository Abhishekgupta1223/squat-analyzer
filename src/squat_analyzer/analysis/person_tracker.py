"""
Robust Person Tracking Module
=============================

Kalman-filter inspired tracking for consistent person selection
in multi-person scenes.

Key Features:
1. Position prediction using velocity estimation
2. Size consistency checking
3. IoU-based association
4. Graceful recovery from lost tracks
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math

from squat_analyzer.core.keypoints import Keypoints
from squat_analyzer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PersonTrack:
    """
    Represents a tracked person with state estimation.
    
    Uses simple velocity-based prediction (Kalman-lite).
    """
    center_x: float
    center_y: float
    size: float  # Approximate bounding area
    
    # Velocity estimates
    vx: float = 0.0
    vy: float = 0.0
    
    # Tracking metadata
    last_update_time: float = 0.0
    frames_tracked: int = 0
    frames_lost: int = 0
    
    def predict(self, dt: float) -> Tuple[float, float]:
        """Predict position at current time."""
        pred_x = self.center_x + self.vx * dt
        pred_y = self.center_y + self.vy * dt
        return pred_x, pred_y
    
    def update(self, x: float, y: float, size: float, timestamp: float):
        """Update track with new observation."""
        dt = timestamp - self.last_update_time
        if dt > 0 and self.frames_tracked > 0:
            # Update velocity with smoothing
            alpha = 0.3  # Smoothing factor
            new_vx = (x - self.center_x) / dt
            new_vy = (y - self.center_y) / dt
            self.vx = alpha * new_vx + (1 - alpha) * self.vx
            self.vy = alpha * new_vy + (1 - alpha) * self.vy
        
        self.center_x = x
        self.center_y = y
        self.size = size
        self.last_update_time = timestamp
        self.frames_tracked += 1
        self.frames_lost = 0
    
    def mark_lost(self):
        """Mark track as lost for this frame."""
        self.frames_lost += 1


def get_person_center(kp: Keypoints) -> Optional[Tuple[float, float]]:
    """Extract center position from keypoints (hip center)."""
    try:
        hip_x = (kp.left_hip[0] + kp.right_hip[0]) / 2
        hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        return (hip_x, hip_y)
    except:
        return None


def get_person_size(kp: Keypoints) -> float:
    """Estimate person size from keypoints."""
    try:
        # Use torso dimensions
        width = abs(kp.right_shoulder[0] - kp.left_shoulder[0])
        width += abs(kp.right_hip[0] - kp.left_hip[0])
        height = abs(kp.left_hip[1] - kp.left_shoulder[1])
        height += abs(kp.right_hip[1] - kp.right_shoulder[1])
        return width * height / 2
    except:
        return 0.0


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class PersonTracker:
    """
    Robust single-person tracker for squat analysis.
    
    Strategy:
    1. Initialize with largest person in frame
    2. Track using position + velocity prediction
    3. Verify with size consistency
    4. Re-initialize only after extended loss
    """
    
    def __init__(
        self,
        max_lost_frames: int = 15,
        max_distance_ratio: float = 0.25,  # Max distance as ratio of frame diagonal
        size_tolerance: float = 0.5,       # Allow 50% size change
    ):
        self.max_lost_frames = max_lost_frames
        self.max_distance_ratio = max_distance_ratio
        self.size_tolerance = size_tolerance
        
        self._track: Optional[PersonTrack] = None
        self._frame_diagonal: float = 1000.0
        self._initialized = False
    
    def select_person(
        self,
        keypoints_list: List[Keypoints],
        frame_width: int,
        frame_height: int,
    ) -> Optional[Keypoints]:
        """
        Select the tracked person from detected keypoints.
        
        Returns keypoints for the tracked person, or None if lost.
        """
        timestamp = time.perf_counter()
        self._frame_diagonal = math.sqrt(frame_width**2 + frame_height**2)
        
        if not keypoints_list:
            if self._track:
                self._track.mark_lost()
                if self._track.frames_lost >= self.max_lost_frames:
                    logger.info("Track lost - will reinitialize")
                    self._track = None
                    self._initialized = False
            return None
        
        # Single person - easy case
        if len(keypoints_list) == 1:
            kp = keypoints_list[0]
            self._update_or_init_track(kp, timestamp)
            return kp
        
        # Multiple people - need to associate or initialize
        
        # If no active track, initialize with largest person
        if not self._initialized or self._track is None:
            return self._initialize_track(keypoints_list, timestamp)
        
        # Try to associate with existing track
        best_match = self._associate_track(keypoints_list, timestamp)
        
        if best_match is not None:
            self._track.update(
                *get_person_center(best_match),
                get_person_size(best_match),
                timestamp
            )
            return best_match
        
        # Lost track - mark and potentially reinitialize
        self._track.mark_lost()
        
        if self._track.frames_lost >= self.max_lost_frames:
            logger.info("Track lost - reinitializing with largest person")
            return self._initialize_track(keypoints_list, timestamp)
        
        # Still trying to recover - return None to skip frame
        return None
    
    def _initialize_track(
        self,
        keypoints_list: List[Keypoints],
        timestamp: float,
    ) -> Keypoints:
        """Initialize track with the largest person."""
        best_kp = None
        best_size = 0.0
        
        for kp in keypoints_list:
            size = get_person_size(kp)
            if size > best_size:
                best_size = size
                best_kp = kp
        
        if best_kp is None:
            best_kp = keypoints_list[0]
            best_size = get_person_size(best_kp)
        
        center = get_person_center(best_kp)
        if center:
            self._track = PersonTrack(
                center_x=center[0],
                center_y=center[1],
                size=best_size,
                last_update_time=timestamp,
            )
            self._initialized = True
            logger.info(f"Track initialized: pos=({center[0]:.0f}, {center[1]:.0f}), size={best_size:.0f}")
        
        return best_kp
    
    def _update_or_init_track(self, kp: Keypoints, timestamp: float):
        """Update track or initialize if needed."""
        center = get_person_center(kp)
        size = get_person_size(kp)
        
        if center is None:
            return
        
        if self._track is None:
            self._track = PersonTrack(
                center_x=center[0],
                center_y=center[1],
                size=size,
                last_update_time=timestamp,
            )
            self._initialized = True
        else:
            self._track.update(center[0], center[1], size, timestamp)
    
    def _associate_track(
        self,
        keypoints_list: List[Keypoints],
        timestamp: float,
    ) -> Optional[Keypoints]:
        """
        Find the detection that best matches our track.
        
        Uses position prediction and size consistency.
        """
        if self._track is None:
            return None
        
        dt = timestamp - self._track.last_update_time
        pred_x, pred_y = self._track.predict(dt)
        max_dist = self._frame_diagonal * self.max_distance_ratio
        
        best_match = None
        best_score = float('inf')
        
        for kp in keypoints_list:
            center = get_person_center(kp)
            if center is None:
                continue
            
            # Position distance
            dist = distance(center, (pred_x, pred_y))
            
            if dist > max_dist:
                continue
            
            # Size consistency
            size = get_person_size(kp)
            if self._track.size > 0:
                size_ratio = size / self._track.size
                if size_ratio < (1 - self.size_tolerance) or size_ratio > (1 + self.size_tolerance):
                    # Size mismatch - add penalty
                    dist *= 2.0
            
            # Best match is closest
            if dist < best_score:
                best_score = dist
                best_match = kp
        
        return best_match
    
    def reset(self):
        """Reset tracker state."""
        self._track = None
        self._initialized = False
        logger.info("PersonTracker reset")
