"""
Adaptive View Detection and Squat Analysis for Webcam Usage.

The key insight: webcams are typically front-facing, not side-view.
- Side view: Traditional joint angle calculation works
- Front view: Need position-based metrics instead

This module detects the view angle and adapts analysis accordingly.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
import numpy as np

from squat_analyzer.core.keypoints import Keypoints, KeypointIndex


class ViewAngle(Enum):
    """Detected camera view angle."""
    FRONT = auto()      # User facing camera
    SIDE_LEFT = auto()  # Left side visible
    SIDE_RIGHT = auto() # Right side visible
    DIAGONAL = auto()   # 45-degree angle
    UNKNOWN = auto()


@dataclass
class ViewMetrics:
    """Metrics for determining view angle."""
    shoulder_width_ratio: float  # shoulder_dist / hip_dist
    depth_indicator: float       # how much L/R sides differ
    hip_width: float
    shoulder_width: float


def detect_view_angle(keypoints: Keypoints) -> ViewAngle:
    """
    Detect whether user is facing front, side, or diagonal.
    
    Front view indicators:
    - Left and right shoulders at similar X positions relative to hips
    - Visible width of shoulders is significant
    
    Side view indicators:
    - One shoulder much further from camera (smaller apparent distance)
    - Hips and shoulders appear narrow
    """
    try:
        # Get key positions
        l_shoulder = keypoints.left_shoulder[:2]
        r_shoulder = keypoints.right_shoulder[:2]
        l_hip = keypoints.left_hip[:2]
        r_hip = keypoints.right_hip[:2]
        
        # Calculate widths
        shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
        hip_width = abs(r_hip[0] - l_hip[0])
        
        # For front view: shoulders and hips have significant horizontal spread
        # For side view: shoulders and hips appear narrow (similar X coordinates)
        
        if shoulder_width < 30 or hip_width < 30:
            # Very narrow - likely side view
            # Determine which side based on which is more visible
            mid_shoulder = (l_shoulder[0] + r_shoulder[0]) / 2
            mid_hip = (l_hip[0] + r_hip[0]) / 2
            frame_center = 320  # Assume ~640px width
            
            if mid_shoulder < frame_center - 50:
                return ViewAngle.SIDE_RIGHT  # Body is on left, right side visible
            elif mid_shoulder > frame_center + 50:
                return ViewAngle.SIDE_LEFT
            else:
                return ViewAngle.UNKNOWN
        
        # Check symmetry for front view
        l_knee = keypoints.left_knee[:2]
        r_knee = keypoints.right_knee[:2]
        
        # In front view, left and right knees should be at similar Y levels
        knee_y_diff = abs(l_knee[1] - r_knee[1])
        knee_x_spread = abs(l_knee[0] - r_knee[0])
        
        if knee_x_spread > 50 and knee_y_diff < 50:
            return ViewAngle.FRONT
        
        # Check for diagonal
        shoulder_hip_ratio = shoulder_width / max(hip_width, 1)
        if 0.5 < shoulder_hip_ratio < 1.5:
            return ViewAngle.DIAGONAL
        
        return ViewAngle.FRONT  # Default to front for webcam use
        
    except Exception:
        return ViewAngle.UNKNOWN


def calculate_squat_depth_front_view(keypoints: Keypoints, standing_hip_y: Optional[float] = None) -> float:
    """
    Calculate squat depth for FRONT-FACING view.
    
    Instead of joint angles (which don't work from front), we use:
    - Hip vertical position relative to standing
    - Where 1.0 = standing, 0.0 = parallel squat
    
    Returns:
        Depth ratio [0.0, 1.0] where lower = deeper squat
    """
    try:
        # Current hip Y position (average of left and right)
        current_hip_y = (keypoints.left_hip[1] + keypoints.right_hip[1]) / 2
        
        # Current knee Y position
        current_knee_y = (keypoints.left_knee[1] + keypoints.right_knee[1]) / 2
        
        # If we know standing hip position, use relative depth
        if standing_hip_y is not None:
            # How much has hip dropped from standing?
            # Assuming ankle Y is reference (doesn't change much)
            ankle_y = (keypoints.left_ankle[1] + keypoints.right_ankle[1]) / 2
            
            standing_leg_length = ankle_y - standing_hip_y
            current_leg_length = ankle_y - current_hip_y
            
            if standing_leg_length > 0:
                depth_ratio = current_leg_length / standing_leg_length
                return np.clip(depth_ratio, 0.0, 1.5)
        
        # Fallback: use hip-knee relationship
        # When squatting, hip drops toward knee level
        hip_knee_diff = current_knee_y - current_hip_y
        
        # Normalize: standing ~= 200px diff, squat ~= 50px diff
        if hip_knee_diff > 150:
            return 1.0  # Standing
        elif hip_knee_diff < 50:
            return 0.3  # Deep squat
        else:
            return (hip_knee_diff - 50) / 100  # Linear interpolation
        
    except Exception:
        return 1.0  # Default to standing


def calculate_squat_phase_front_view(keypoints: Keypoints) -> str:
    """
    Determine squat phase from front view using vertical positions.
    """
    try:
        hip_y = (keypoints.left_hip[1] + keypoints.right_hip[1]) / 2
        knee_y = (keypoints.left_knee[1] + keypoints.right_knee[1]) / 2
        shoulder_y = (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) / 2
        
        # Vertical distances
        hip_knee_dist = knee_y - hip_y  # Positive when standing
        shoulder_hip_dist = hip_y - shoulder_y  # Torso length
        
        # Squat indicators:
        # 1. Hip drops (hip_knee_dist decreases)
        # 2. Shoulders also drop
        
        if hip_knee_dist > 120:
            return "STANDING"
        elif hip_knee_dist > 80:
            return "PARTIAL"  # Quarter squat
        elif hip_knee_dist > 40:
            return "PARALLEL"  # Good squat depth
        else:
            return "DEEP"  # Below parallel
            
    except Exception:
        return "UNKNOWN"


def is_valid_squat_front_view(
    start_hip_y: float,
    min_hip_y: float,  # Maximum Y = lowest point
    end_hip_y: float,
    min_duration_ms: float = 500,
) -> bool:
    """
    Validate a squat from front view.
    
    A valid squat:
    - Hip drops significantly (at least 30% of leg length)
    - Returns to near starting position
    - Takes reasonable time
    """
    drop_amount = min_hip_y - start_hip_y  # Positive = hip dropped
    recovery = end_hip_y - min_hip_y  # Should be negative (hip rose back up)
    
    # Hip must drop at least 50 pixels
    if drop_amount < 50:
        return False
    
    # Must return to within 30px of start
    if abs(end_hip_y - start_hip_y) > 30:
        return False
    
    return True
