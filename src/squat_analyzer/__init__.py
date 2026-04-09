"""
Squat Analyzer - Production-Grade Pose Analysis System
======================================================

A research-backed, real-time squat posture analysis system using
state-of-the-art computer vision and biomechanical analysis.

Features:
    - Real-time pose estimation using YOLOv8-pose
    - 6 research-validated biomechanical rules
    - Adaptive signal filtering (One-Euro Filter)
    - State machine for squat phase detection
    - Priority-based corrective feedback

Example:
    >>> from squat_analyzer import SquatAnalyzer, Settings
    >>> settings = Settings()
    >>> analyzer = SquatAnalyzer(settings)
    >>> analyzer.run()

"""

__version__ = "2.0.0"
__author__ = "CV Engineering Team"
__all__ = [
    "SquatAnalyzer",
    "Settings",
    "PoseEstimator",
    "BiomechanicsEngine",
    "SquatDetector",
    "__version__",
]

from squat_analyzer.config.settings import Settings
from squat_analyzer.core.pose_estimator import PoseEstimator
from squat_analyzer.analysis.biomechanics import BiomechanicsEngine
from squat_analyzer.analysis.squat_detector import SquatDetector
from squat_analyzer.main import SquatAnalyzer
