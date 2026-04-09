"""Test fixtures and configuration for pytest."""

import numpy as np
import pytest

from squat_analyzer.config.settings import Settings
from squat_analyzer.core.keypoints import Keypoints, KeypointIndex


@pytest.fixture
def settings():
    """Create default settings for testing."""
    return Settings()


@pytest.fixture
def sample_keypoints():
    """Create sample keypoints for a standing pose."""
    # Simulated standing pose (17 keypoints)
    # Format: [x, y] for each keypoint
    points = np.array([
        [320, 100],   # nose
        [310, 90],    # left_eye
        [330, 90],    # right_eye
        [300, 95],    # left_ear
        [340, 95],    # right_ear
        [280, 180],   # left_shoulder
        [360, 180],   # right_shoulder
        [250, 280],   # left_elbow
        [390, 280],   # right_elbow
        [230, 380],   # left_wrist
        [410, 380],   # right_wrist
        [290, 350],   # left_hip
        [350, 350],   # right_hip
        [290, 500],   # left_knee
        [350, 500],   # right_knee
        [290, 650],   # left_ankle
        [350, 650],   # right_ankle
    ], dtype=np.float32)
    
    return Keypoints(points=points)


@pytest.fixture
def squat_bottom_keypoints():
    """Create sample keypoints for a squat bottom position."""
    # Simulated deep squat pose
    points = np.array([
        [320, 200],   # nose (lower due to squat)
        [310, 190],   # left_eye
        [330, 190],   # right_eye
        [300, 195],   # left_ear
        [340, 195],   # right_ear
        [280, 280],   # left_shoulder
        [360, 280],   # right_shoulder
        [250, 350],   # left_elbow
        [390, 350],   # right_elbow
        [230, 420],   # left_wrist
        [410, 420],   # right_wrist
        [290, 400],   # left_hip
        [350, 400],   # right_hip
        [280, 500],   # left_knee (forward)
        [360, 500],   # right_knee (forward)
        [290, 600],   # left_ankle
        [350, 600],   # right_ankle
    ], dtype=np.float32)
    
    return Keypoints(points=points)


@pytest.fixture
def knee_valgus_keypoints():
    """Create keypoints with knee valgus (inward collapse)."""
    # Knees pointing inward
    points = np.array([
        [320, 200],   # nose
        [310, 190],   # left_eye
        [330, 190],   # right_eye
        [300, 195],   # left_ear
        [340, 195],   # right_ear
        [280, 280],   # left_shoulder
        [360, 280],   # right_shoulder
        [250, 350],   # left_elbow
        [390, 350],   # right_elbow
        [230, 420],   # left_wrist
        [410, 420],   # right_wrist
        [290, 400],   # left_hip
        [350, 400],   # right_hip
        [315, 500],   # left_knee - moved inward
        [325, 500],   # right_knee - moved inward
        [290, 600],   # left_ankle
        [350, 600],   # right_ankle
    ], dtype=np.float32)
    
    return Keypoints(points=points)
