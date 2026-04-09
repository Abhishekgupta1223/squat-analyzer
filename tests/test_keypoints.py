"""Tests for keypoints module."""

import numpy as np
import pytest

from squat_analyzer.core.keypoints import (
    Keypoints,
    KeypointIndex,
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
    SQUAT_CRITICAL_KEYPOINTS,
)


class TestKeypoints:
    """Tests for Keypoints class."""
    
    def test_keypoints_creation(self, sample_keypoints):
        """Test basic keypoint creation."""
        assert sample_keypoints.points.shape == (17, 2)
    
    def test_keypoints_with_confidence(self):
        """Test keypoints with embedded confidence."""
        # Shape (17, 3) with x, y, confidence
        points_with_conf = np.random.rand(17, 3).astype(np.float32)
        kps = Keypoints(points=points_with_conf)
        
        assert kps.points.shape == (17, 2)
        assert kps.confidence is not None
        assert len(kps.confidence) == 17
    
    def test_keypoint_access(self, sample_keypoints):
        """Test accessing individual keypoints."""
        nose = sample_keypoints.nose
        assert nose[0] == 320
        assert nose[1] == 100
        
        left_knee = sample_keypoints.left_knee
        assert len(left_knee) == 2
    
    def test_midpoint_computation(self, sample_keypoints):
        """Test midpoint properties."""
        mid_hip = sample_keypoints.mid_hip
        assert len(mid_hip) == 2
        
        # Should be average of left and right hip
        expected_x = (290 + 350) / 2
        assert mid_hip[0] == expected_x
    
    def test_keypoint_index_access(self, sample_keypoints):
        """Test access via KeypointIndex enum."""
        nose = sample_keypoints[KeypointIndex.NOSE]
        assert np.array_equal(nose, sample_keypoints.nose)
    
    def test_invalid_shape(self):
        """Test that invalid shapes raise errors."""
        with pytest.raises(ValueError):
            Keypoints(points=np.zeros((10, 2)))
    
    def test_confidence_check(self):
        """Test confidence threshold checking."""
        points = np.ones((17, 3), dtype=np.float32)
        points[:, 2] = 0.8  # High confidence
        points[0, 2] = 0.3  # Low confidence for nose
        
        kps = Keypoints(points=points)
        assert kps.is_visible(0, threshold=0.5) is False
        assert kps.is_visible(1, threshold=0.5) is True
    
    def test_has_critical_keypoints(self, sample_keypoints):
        """Test critical keypoint check."""
        # With default confidence (1.0), all should be visible
        sample_keypoints.confidence = np.ones(17, dtype=np.float32)
        assert sample_keypoints.has_critical_keypoints()
    
    def test_to_dict(self, sample_keypoints):
        """Test dictionary conversion."""
        kps_dict = sample_keypoints.to_dict()
        assert "nose" in kps_dict
        assert "left_knee" in kps_dict
        assert len(kps_dict["nose"]) == 2


class TestKeypointConstants:
    """Tests for keypoint constants."""
    
    def test_keypoint_names_count(self):
        """Test correct number of keypoint names."""
        assert len(KEYPOINT_NAMES) == 17
    
    def test_skeleton_connections(self):
        """Test skeleton connections are valid."""
        for i, j in SKELETON_CONNECTIONS:
            assert 0 <= i < 17
            assert 0 <= j < 17
    
    def test_critical_keypoints(self):
        """Test critical keypoints include necessary joints."""
        critical = set(SQUAT_CRITICAL_KEYPOINTS)
        assert KeypointIndex.LEFT_HIP in critical
        assert KeypointIndex.RIGHT_KNEE in critical
        assert KeypointIndex.LEFT_ANKLE in critical
