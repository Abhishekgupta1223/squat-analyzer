"""
Angle Calculation Module for Biomechanical Analysis
====================================================

Provides precise angle calculations between body segments using
vector mathematics. Supports both 2D and 3D computations with
proper handling of edge cases.

Mathematical Foundation:
    - Angles computed using dot product: cos(θ) = (a·b) / (|a||b|)
    - Results in range [0°, 180°] for joint angles
    - Signed angles use cross product for direction
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from squat_analyzer.core.keypoints import Keypoints, KeypointIndex
from squat_analyzer.utils.logging import get_logger

logger = get_logger(__name__)

# Type aliases
Point2D = NDArray[np.float32]  # Shape: (2,)
Point3D = NDArray[np.float32]  # Shape: (3,)
Point = Union[Point2D, Point3D]


class AngleCalculator:
    """
    Calculate angles between body segments for biomechanical analysis.
    
    This class provides methods to compute:
        - Joint angles (e.g., knee flexion, elbow angle)
        - Segment angles relative to vertical/horizontal
        - Deviation angles for valgus/varus detection
    
    All angles are returned in degrees unless otherwise specified.
    
    Example:
        >>> calc = AngleCalculator()
        >>> knee_angle = calc.joint_angle(hip, knee, ankle)
        >>> print(f"Knee flexion: {knee_angle:.1f}°")
    """
    
    @staticmethod
    def _normalize(vector: Point) -> Point:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return np.zeros_like(vector)
        return vector / norm
    
    @staticmethod
    def _angle_between_vectors(v1: Point, v2: Point) -> float:
        """
        Calculate angle between two vectors using dot product.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Angle in degrees [0°, 180°], or 180.0 on error
        """
        try:
            # Check for NaN or zero vectors
            if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
                return 180.0
            
            v1_norm = AngleCalculator._normalize(v1)
            v2_norm = AngleCalculator._normalize(v2)
            
            # Check for zero vectors after normalization
            if np.allclose(v1_norm, 0) or np.allclose(v2_norm, 0):
                return 180.0
            
            # Clamp dot product to [-1, 1] to avoid numerical issues
            dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            
            result = np.degrees(np.arccos(dot))
            return result if not np.isnan(result) else 180.0
        except Exception:
            return 180.0
    
    def joint_angle(
        self,
        point_a: Point,
        point_b: Point,  # Joint point
        point_c: Point,
    ) -> float:
        """
        Calculate angle at a joint (point_b) formed by segments AB and BC.
        
        The angle represents the opening at the joint, where:
            - 180° = fully extended (straight)
            - 0° = fully flexed (folded)
        
        Args:
            point_a: First endpoint (e.g., hip)
            point_b: Joint vertex (e.g., knee)
            point_c: Second endpoint (e.g., ankle)
            
        Returns:
            Joint angle in degrees [0°, 180°]
            
        Example:
            >>> # Knee angle with hip-knee-ankle
            >>> angle = calc.joint_angle(hip, knee, ankle)
        """
        # Vectors from joint to endpoints
        ba = np.array(point_a) - np.array(point_b)
        bc = np.array(point_c) - np.array(point_b)
        
        return self._angle_between_vectors(ba, bc)
    
    def angle_with_vertical(self, point_a: Point, point_b: Point) -> float:
        """
        Calculate angle of segment AB with respect to vertical axis.
        
        Vertical is defined as pointing upward (negative y in image coords).
        
        Args:
            point_a: Start point of segment
            point_b: End point of segment
            
        Returns:
            Angle from vertical in degrees [0°, 180°]
            
        Example:
            >>> # Torso inclination (shoulder to hip)
            >>> lean = calc.angle_with_vertical(shoulder, hip)
        """
        segment = np.array(point_b) - np.array(point_a)
        
        # Vertical vector (pointing up in image coordinates)
        vertical = np.array([0, -1] if len(segment) == 2 else [0, -1, 0])
        
        return self._angle_between_vectors(segment, vertical)
    
    def angle_with_horizontal(self, point_a: Point, point_b: Point) -> float:
        """
        Calculate angle of segment AB with respect to horizontal axis.
        
        Args:
            point_a: Start point of segment
            point_b: End point of segment
            
        Returns:
            Angle from horizontal in degrees [0°, 180°]
        """
        segment = np.array(point_b) - np.array(point_a)
        
        # Horizontal vector (pointing right)
        horizontal = np.array([1, 0] if len(segment) == 2 else [1, 0, 0])
        
        return self._angle_between_vectors(segment, horizontal)
    
    def signed_angle_2d(
        self,
        point_a: Point2D,
        point_b: Point2D,
        point_c: Point2D,
    ) -> float:
        """
        Calculate signed angle at joint B, positive = counterclockwise.
        
        Uses cross product to determine rotation direction.
        
        Args:
            point_a: First endpoint
            point_b: Joint vertex
            point_c: Second endpoint
            
        Returns:
            Signed angle in degrees [-180°, 180°]
        """
        ba = np.array(point_a, dtype=np.float64)[:2] - np.array(point_b, dtype=np.float64)[:2]
        bc = np.array(point_c, dtype=np.float64)[:2] - np.array(point_b, dtype=np.float64)[:2]
        
        # 2D cross product gives signed area (sin * |ba| * |bc|)
        cross = ba[0] * bc[1] - ba[1] * bc[0]
        dot = np.dot(ba, bc)
        
        return np.degrees(np.arctan2(cross, dot))
    
    # =========================================================================
    # Squat-Specific Calculations
    # =========================================================================
    
    def knee_flexion_angle(self, keypoints: Keypoints, side: str = "both") -> float:
        """
        Calculate knee flexion angle.
        
        Args:
            keypoints: Detected keypoints
            side: 'left', 'right', or 'both' (average)
            
        Returns:
            Knee angle in degrees (180° = straight, ~70-90° = deep squat)
        """
        if side == "left":
            return self.joint_angle(
                keypoints.left_hip,
                keypoints.left_knee,
                keypoints.left_ankle,
            )
        elif side == "right":
            return self.joint_angle(
                keypoints.right_hip,
                keypoints.right_knee,
                keypoints.right_ankle,
            )
        else:
            left = self.joint_angle(
                keypoints.left_hip,
                keypoints.left_knee,
                keypoints.left_ankle,
            )
            right = self.joint_angle(
                keypoints.right_hip,
                keypoints.right_knee,
                keypoints.right_ankle,
            )
            return (left + right) / 2
    
    def hip_flexion_angle(self, keypoints: Keypoints, side: str = "both") -> float:
        """
        Calculate hip flexion angle (shoulder-hip-knee).
        
        Args:
            keypoints: Detected keypoints
            side: 'left', 'right', or 'both' (average)
            
        Returns:
            Hip angle in degrees (180° = standing tall, ~90° = deep squat)
        """
        if side == "left":
            return self.joint_angle(
                keypoints.left_shoulder,
                keypoints.left_hip,
                keypoints.left_knee,
            )
        elif side == "right":
            return self.joint_angle(
                keypoints.right_shoulder,
                keypoints.right_hip,
                keypoints.right_knee,
            )
        else:
            left = self.joint_angle(
                keypoints.left_shoulder,
                keypoints.left_hip,
                keypoints.left_knee,
            )
            right = self.joint_angle(
                keypoints.right_shoulder,
                keypoints.right_hip,
                keypoints.right_knee,
            )
            return (left + right) / 2
    
    def torso_inclination(self, keypoints: Keypoints) -> float:
        """
        Calculate torso forward lean angle from vertical.
        
        Uses shoulder and hip midpoints to determine torso vector.
        
        Args:
            keypoints: Detected keypoints
            
        Returns:
            Forward lean angle in degrees (0° = upright, >45° = leaning forward)
        """
        # Torso segment: from shoulder (top) to hip (bottom)
        segment = np.array(keypoints.mid_hip) - np.array(keypoints.mid_shoulder)
        
        # In image coordinates, Y increases downward
        # Vertical DOWN direction = [0, 1] (positive Y)
        # When standing upright, torso points DOWN so angle with [0,1] should be ~0°
        vertical_down = np.array([0, 1])
        
        return self._angle_between_vectors(segment, vertical_down)
    
    def knee_valgus_angle(self, keypoints: Keypoints, side: str = "both") -> float:
        """
        Estimate knee valgus (inward collapse) angle.
        
        Compares knee position relative to the hip-ankle line in frontal plane.
        Positive values indicate valgus (knees inward).
        
        This is an approximation from 2D - true valgus requires 3D analysis.
        
        Args:
            keypoints: Detected keypoints
            side: 'left', 'right', or 'both' (max of both)
            
        Returns:
            Estimated valgus angle in degrees (>10° is concerning)
        """
        def calc_side_valgus(hip: Point2D, knee: Point2D, ankle: Point2D) -> float:
            # Vector from ankle to hip (reference line)
            hip_ankle = np.array(hip) - np.array(ankle)
            # Vector from ankle to knee
            knee_pos = np.array(knee) - np.array(ankle)
            
            # Project knee onto hip-ankle line direction
            hip_ankle_norm = self._normalize(hip_ankle)
            projection_length = np.dot(knee_pos, hip_ankle_norm)
            projection = projection_length * hip_ankle_norm
            
            # Perpendicular deviation (medial/lateral)
            deviation = knee_pos - projection
            
            # Compute deviation angle
            deviation_mag = np.linalg.norm(deviation)
            reference_length = np.linalg.norm(hip_ankle)
            
            if reference_length < 1e-8:
                return 0.0
            
            # Angle from ratio (small angle approximation valid for valgus)
            return np.degrees(np.arctan2(deviation_mag, projection_length))
        
        if side == "left":
            return calc_side_valgus(
                keypoints.left_hip,
                keypoints.left_knee,
                keypoints.left_ankle,
            )
        elif side == "right":
            return calc_side_valgus(
                keypoints.right_hip,
                keypoints.right_knee,
                keypoints.right_ankle,
            )
        else:
            left = calc_side_valgus(
                keypoints.left_hip,
                keypoints.left_knee,
                keypoints.left_ankle,
            )
            right = calc_side_valgus(
                keypoints.right_hip,
                keypoints.right_knee,
                keypoints.right_ankle,
            )
            return max(left, right)
    
    def knee_over_toe_ratio(self, keypoints: Keypoints, side: str = "both") -> float:
        """
        Calculate how far the knee extends past the toes.
        
        Returns ratio of knee extension beyond ankle relative to foot length.
        Positive = knee past ankle, negative = knee behind ankle.
        
        Args:
            keypoints: Detected keypoints
            side: 'left', 'right', or 'both' (max of both)
            
        Returns:
            Ratio (0.15 = 15% past toes, which is our threshold)
        """
        def calc_side_ratio(knee: Point2D, ankle: Point2D) -> float:
            # In side view, x-coordinate difference
            knee_x = knee[0]
            ankle_x = ankle[0]
            
            # Estimate foot length as ~20% of leg height
            leg_height = abs(knee[1] - ankle[1])
            est_foot_length = max(leg_height * 0.3, 1.0)
            
            # How far knee extends past ankle
            extension = knee_x - ankle_x
            
            return extension / est_foot_length
        
        if side == "left":
            return calc_side_ratio(keypoints.left_knee, keypoints.left_ankle)
        elif side == "right":
            return calc_side_ratio(keypoints.right_knee, keypoints.right_ankle)
        else:
            left = calc_side_ratio(keypoints.left_knee, keypoints.left_ankle)
            right = calc_side_ratio(keypoints.right_knee, keypoints.right_ankle)
            return max(abs(left), abs(right))
    
    def thigh_to_horizontal_angle(self, keypoints: Keypoints, side: str = "both") -> float:
        """
        Calculate angle of thigh relative to horizontal (for depth check).
        
        At parallel: ~0°
        Above parallel: positive
        Below parallel (ATG): negative
        
        Args:
            keypoints: Detected keypoints
            side: 'left', 'right', or 'both' (average)
            
        Returns:
            Angle from horizontal in degrees
        """
        def calc_thigh_angle(hip: Point2D, knee: Point2D) -> float:
            angle = self.angle_with_horizontal(hip, knee)
            # Convert to signed: below horizontal = negative
            if knee[1] > hip[1]:  # Knee lower in image coords
                return angle - 90
            return 90 - angle
        
        if side == "left":
            return calc_thigh_angle(keypoints.left_hip, keypoints.left_knee)
        elif side == "right":
            return calc_thigh_angle(keypoints.right_hip, keypoints.right_knee)
        else:
            left = calc_thigh_angle(keypoints.left_hip, keypoints.left_knee)
            right = calc_thigh_angle(keypoints.right_hip, keypoints.right_knee)
            return (left + right) / 2
    
    def compute_all_angles(self, keypoints: Keypoints) -> dict[str, float]:
        """
        Compute all squat-relevant angles at once.
        
        Args:
            keypoints: Detected keypoints
            
        Returns:
            Dictionary of all computed angles
        """
        return {
            "knee_flexion": self.knee_flexion_angle(keypoints),
            "knee_flexion_left": self.knee_flexion_angle(keypoints, "left"),
            "knee_flexion_right": self.knee_flexion_angle(keypoints, "right"),
            "hip_flexion": self.hip_flexion_angle(keypoints),
            "hip_flexion_left": self.hip_flexion_angle(keypoints, "left"),
            "hip_flexion_right": self.hip_flexion_angle(keypoints, "right"),
            "torso_inclination": self.torso_inclination(keypoints),
            "knee_valgus": self.knee_valgus_angle(keypoints),
            "knee_valgus_left": self.knee_valgus_angle(keypoints, "left"),
            "knee_valgus_right": self.knee_valgus_angle(keypoints, "right"),
            "knee_over_toe": self.knee_over_toe_ratio(keypoints),
            "thigh_angle": self.thigh_to_horizontal_angle(keypoints),
        }
