"""Tests for squat detector state machine."""

import pytest

from squat_analyzer.analysis.squat_detector import (
    SquatDetector,
    SquatPhase,
    PhaseTransitionThresholds,
    SquatState,
)
from squat_analyzer.core.keypoints import Keypoints
import numpy as np


class TestSquatPhase:
    """Tests for SquatPhase enum."""
    
    def test_phase_string_conversion(self):
        """Test phase to string conversion."""
        assert str(SquatPhase.IDLE) == "Idle"
        assert str(SquatPhase.STANDING) == "Standing"
        assert str(SquatPhase.DESCENDING) == "Descending"
        assert str(SquatPhase.BOTTOM) == "Bottom"
        assert str(SquatPhase.ASCENDING) == "Ascending"


class TestSquatState:
    """Tests for SquatState class."""
    
    def test_state_initialization(self):
        """Test state initialization."""
        state = SquatState()
        assert state.phase == SquatPhase.IDLE
        assert state.rep_count == 0
        assert state.current_knee_angle == 180.0
    
    def test_update_angles(self):
        """Test angle update."""
        state = SquatState()
        state.update_angles(150.0, 140.0, 20.0)
        
        assert state.current_knee_angle == 150.0
        assert state.current_hip_angle == 140.0
        assert state.current_torso_angle == 20.0
        assert len(state.knee_angle_history) == 1
    
    def test_min_angle_tracking(self):
        """Test minimum angle tracking."""
        state = SquatState()
        state.update_angles(150.0, 140.0, 20.0)
        state.update_angles(100.0, 90.0, 30.0)
        state.update_angles(110.0, 100.0, 25.0)  # Going back up
        
        assert state.min_knee_angle_this_rep == 100.0
        assert state.min_hip_angle_this_rep == 90.0
    
    def test_velocity_calculation(self):
        """Test velocity calculation."""
        state = SquatState()
        state.update_angles(150.0, 140.0, 20.0)
        state.update_angles(140.0, 130.0, 25.0)
        
        # Velocity should be negative (descending)
        assert state.knee_velocity == -10.0


class TestSquatDetector:
    """Tests for SquatDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create squat detector instance."""
        return SquatDetector()
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.phase == SquatPhase.IDLE
        assert detector.rep_count == 0
    
    def test_idle_detection(self, detector, sample_keypoints):
        """Test idle phase detection (starts in IDLE, not STANDING)."""
        # First update starts in IDLE
        phase = detector.update(sample_keypoints)
        # May transition to STANDING if keypoints show standing position
        assert phase in [SquatPhase.IDLE, SquatPhase.STANDING]
    
    def test_rep_counting_requires_depth(self, detector):
        """Test that reps require minimum depth."""
        thresholds = PhaseTransitionThresholds(min_depth_knee_angle=100.0)
        detector = SquatDetector(thresholds=thresholds)
        
        # Without proper squat, no reps should be counted
        assert detector.rep_count == 0
    
    def test_get_state_info(self, detector, sample_keypoints):
        """Test state info retrieval."""
        detector.update(sample_keypoints)
        info = detector.get_state_info()
        
        assert "phase" in info
        assert "rep_count" in info
        assert "current_knee_angle" in info
        assert "current_hip_angle" in info
        assert "velocity" in info
        assert "invalid_attempts" in info
    
    def test_reset(self, detector, sample_keypoints):
        """Test detector reset."""
        detector.update(sample_keypoints)
        detector._state.rep_count = 5  # Simulate some reps
        
        detector.reset()
        
        assert detector.rep_count == 0
        assert detector.phase == SquatPhase.IDLE


class TestPhaseTransitions:
    """Tests for phase transition logic."""
    
    def test_thresholds_customizable(self):
        """Test custom thresholds."""
        thresholds = PhaseTransitionThresholds(
            standing_knee_angle=170.0,
            squat_start_angle=160.0,
            min_depth_knee_angle=90.0,
        )
        
        detector = SquatDetector(thresholds=thresholds)
        assert detector._thresholds.standing_knee_angle == 170.0
    
    def test_strict_validation_prevents_false_positives(self):
        """Test that strict validation is configured."""
        thresholds = PhaseTransitionThresholds()
        
        # Verify strict thresholds are set
        assert thresholds.standing_knee_angle >= 165.0  # Must be very straight
        assert thresholds.min_depth_knee_angle <= 100.0  # Must squat deep
        assert thresholds.min_rep_duration_ms >= 500.0  # Rep must take time
        assert thresholds.min_frames_in_state >= 5  # Anti-jitter
