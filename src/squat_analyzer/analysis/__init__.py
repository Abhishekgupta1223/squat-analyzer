"""Analysis package for biomechanical rules and squat detection."""

from squat_analyzer.analysis.biomechanics import BiomechanicsEngine
from squat_analyzer.analysis.squat_detector_v2 import SquatDetectorV2 as SquatDetector, SquatPhase
from squat_analyzer.analysis.person_tracker import PersonTracker
from squat_analyzer.analysis.feedback import FeedbackGenerator, FeedbackMessage

__all__ = [
    "BiomechanicsEngine",
    "SquatDetector",
    "SquatPhase",
    "PersonTracker",
    "FeedbackGenerator",
    "FeedbackMessage",
]
