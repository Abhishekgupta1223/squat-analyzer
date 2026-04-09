"""
Keypoint Definitions and Data Structures
=========================================

Defines the 17-keypoint COCO pose format used by YOLOv8-pose,
along with skeleton connections for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


class KeypointIndex(IntEnum):
    """
    COCO keypoint indices for YOLOv8-pose model.
    
    The COCO format defines 17 keypoints covering the full body,
    which is the standard output from YOLOv8-pose models.
    """
    
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Human-readable keypoint names
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


# Skeleton connections for visualization (pairs of keypoint indices)
SKELETON_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Face
    (KeypointIndex.LEFT_EYE, KeypointIndex.NOSE),
    (KeypointIndex.RIGHT_EYE, KeypointIndex.NOSE),
    (KeypointIndex.LEFT_EYE, KeypointIndex.LEFT_EAR),
    (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EAR),
    # Arms
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
    (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
    (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
    # Torso
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_HIP),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP),
    # Legs
    (KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE),
    (KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_ANKLE),
    (KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE),
    (KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_ANKLE),
)


# Keypoints critical for squat analysis
SQUAT_CRITICAL_KEYPOINTS: tuple[KeypointIndex, ...] = (
    KeypointIndex.LEFT_SHOULDER,
    KeypointIndex.RIGHT_SHOULDER,
    KeypointIndex.LEFT_HIP,
    KeypointIndex.RIGHT_HIP,
    KeypointIndex.LEFT_KNEE,
    KeypointIndex.RIGHT_KNEE,
    KeypointIndex.LEFT_ANKLE,
    KeypointIndex.RIGHT_ANKLE,
)


@dataclass
class Keypoints:
    """
    Container for detected keypoints from pose estimation.
    
    Provides convenient access to individual keypoints and computed
    midpoints useful for squat analysis.
    
    Attributes:
        points: Array of shape (17, 2) or (17, 3) containing [x, y] or [x, y, confidence]
        confidence: Optional confidence scores for each keypoint
        
    Example:
        >>> kps = Keypoints(points=detection_result)
        >>> left_knee = kps.left_knee
        >>> hip_midpoint = kps.mid_hip
    """
    
    points: NDArray[np.float32]
    confidence: Optional[NDArray[np.float32]] = None
    
    def __post_init__(self) -> None:
        """Validate and process keypoint data."""
        self.points = np.asarray(self.points, dtype=np.float32)
        
        # Handle (17, 3) format with embedded confidence
        if self.points.shape[-1] == 3:
            if self.confidence is None:
                self.confidence = self.points[:, 2].copy()
            self.points = self.points[:, :2].copy()
        
        # Validate shape
        if self.points.shape != (17, 2):
            raise ValueError(f"Expected shape (17, 2), got {self.points.shape}")
    
    def __getitem__(self, idx: int | KeypointIndex) -> NDArray[np.float32]:
        """Get keypoint by index."""
        return self.points[int(idx)]
    
    def get_confidence(self, idx: int | KeypointIndex) -> float:
        """Get confidence score for a specific keypoint."""
        if self.confidence is None:
            return 1.0
        return float(self.confidence[int(idx)])
    
    def is_visible(self, idx: int | KeypointIndex, threshold: float = 0.5) -> bool:
        """Check if a keypoint is visible (above confidence threshold)."""
        return self.get_confidence(idx) >= threshold
    
    # Individual keypoint properties
    @property
    def nose(self) -> NDArray[np.float32]:
        return self[KeypointIndex.NOSE]
    
    @property
    def left_eye(self) -> NDArray[np.float32]:
        return self[KeypointIndex.LEFT_EYE]
    
    @property
    def right_eye(self) -> NDArray[np.float32]:
        return self[KeypointIndex.RIGHT_EYE]
    
    @property
    def left_ear(self) -> NDArray[np.float32]:
        return self[KeypointIndex.LEFT_EAR]
    
    @property
    def right_ear(self) -> NDArray[np.float32]:
        return self[KeypointIndex.RIGHT_EAR]
    
    @property
    def left_shoulder(self) -> NDArray[np.float32]:
        return self[KeypointIndex.LEFT_SHOULDER]
    
    @property
    def right_shoulder(self) -> NDArray[np.float32]:
        return self[KeypointIndex.RIGHT_SHOULDER]
    
    @property
    def left_elbow(self) -> NDArray[np.float32]:
        return self[KeypointIndex.LEFT_ELBOW]
    
    @property
    def right_elbow(self) -> NDArray[np.float32]:
        return self[KeypointIndex.RIGHT_ELBOW]
    
    @property
    def left_wrist(self) -> NDArray[np.float32]:
        return self[KeypointIndex.LEFT_WRIST]
    
    @property
    def right_wrist(self) -> NDArray[np.float32]:
        return self[KeypointIndex.RIGHT_WRIST]
    
    @property
    def left_hip(self) -> NDArray[np.float32]:
        return self[KeypointIndex.LEFT_HIP]
    
    @property
    def right_hip(self) -> NDArray[np.float32]:
        return self[KeypointIndex.RIGHT_HIP]
    
    @property
    def left_knee(self) -> NDArray[np.float32]:
        return self[KeypointIndex.LEFT_KNEE]
    
    @property
    def right_knee(self) -> NDArray[np.float32]:
        return self[KeypointIndex.RIGHT_KNEE]
    
    @property
    def left_ankle(self) -> NDArray[np.float32]:
        return self[KeypointIndex.LEFT_ANKLE]
    
    @property
    def right_ankle(self) -> NDArray[np.float32]:
        return self[KeypointIndex.RIGHT_ANKLE]
    
    # Computed midpoints for analysis
    @property
    def mid_shoulder(self) -> NDArray[np.float32]:
        """Midpoint between shoulders."""
        return (self.left_shoulder + self.right_shoulder) / 2
    
    @property
    def mid_hip(self) -> NDArray[np.float32]:
        """Midpoint between hips (pelvis center)."""
        return (self.left_hip + self.right_hip) / 2
    
    @property
    def mid_knee(self) -> NDArray[np.float32]:
        """Midpoint between knees."""
        return (self.left_knee + self.right_knee) / 2
    
    @property
    def mid_ankle(self) -> NDArray[np.float32]:
        """Midpoint between ankles."""
        return (self.left_ankle + self.right_ankle) / 2
    
    def has_critical_keypoints(self, threshold: float = 0.5) -> bool:
        """
        Check if enough keypoints critical for squat analysis are visible.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            True if at least 6 of 8 critical keypoints are visible
        """
        visible_count = sum(
            1 for kp in SQUAT_CRITICAL_KEYPOINTS
            if self.is_visible(kp, threshold)
        )
        return visible_count >= 6  # Allow 2 missing keypoints
    
    def has_partial_keypoints(self, min_count: int = 4, threshold: float = 0.05) -> bool:
        """
        Check if we have at least some keypoints for partial analysis.
        
        More lenient than has_critical_keypoints for real-world videos
        where quality varies and full visibility isn't always possible.
        
        Args:
            min_count: Minimum number of visible keypoints required
            threshold: Very low confidence threshold
            
        Returns:
            True if enough keypoints are visible
        """
        visible_count = sum(
            1 for kp in SQUAT_CRITICAL_KEYPOINTS
            if self.is_visible(kp, threshold)
        )
        return visible_count >= min_count
    
    def to_dict(self) -> dict[str, list[float]]:
        """Convert keypoints to a dictionary format."""
        return {
            name: self[i].tolist()
            for i, name in enumerate(KEYPOINT_NAMES)
        }
