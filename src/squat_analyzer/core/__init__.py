"""Core package for pose estimation and angle calculations."""

from squat_analyzer.core.pose_estimator import PoseEstimator
from squat_analyzer.core.keypoints import Keypoints, KEYPOINT_NAMES
from squat_analyzer.core.angles import AngleCalculator

__all__ = ["PoseEstimator", "Keypoints", "KEYPOINT_NAMES", "AngleCalculator"]
