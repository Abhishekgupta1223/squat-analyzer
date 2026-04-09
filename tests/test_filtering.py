"""Tests for signal filtering module."""

import numpy as np
import pytest
import time

from squat_analyzer.filtering.one_euro import (
    OneEuroFilter,
    KeypointFilter,
    LowPassFilter,
)


class TestLowPassFilter:
    """Tests for LowPassFilter class."""
    
    def test_no_filtering(self):
        """Test with alpha=1 (no filtering)."""
        lpf = LowPassFilter(alpha=1.0)
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            result = lpf.filter(v)
            assert result == v
    
    def test_full_filtering(self):
        """Test with alpha=0 (keep first value)."""
        lpf = LowPassFilter(alpha=0.0)
        
        lpf.filter(10.0)  # First value
        assert lpf.filter(20.0) == 10.0
        assert lpf.filter(30.0) == 10.0
    
    def test_smoothing(self):
        """Test that filter smooths noisy signal."""
        lpf = LowPassFilter(alpha=0.5)
        
        lpf.filter(0.0)
        result = lpf.filter(10.0)
        
        # Should be halfway between
        assert result == 5.0
    
    def test_reset(self):
        """Test filter reset."""
        lpf = LowPassFilter(alpha=0.5)
        lpf.filter(100.0)
        lpf.reset()
        
        # After reset, first value should pass through
        assert lpf.filter(50.0) == 50.0


class TestOneEuroFilter:
    """Tests for OneEuroFilter class."""
    
    def test_initialization(self):
        """Test filter initialization."""
        oef = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        assert oef.min_cutoff == 1.0
        assert oef.beta == 0.007
    
    def test_first_value_passthrough(self):
        """Test that first value passes through."""
        oef = OneEuroFilter()
        result = oef.filter(100.0)
        assert result == 100.0
    
    def test_smoothing_slow_signal(self):
        """Test smoothing of slow-changing signal."""
        oef = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        
        values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        results = []
        
        for i, v in enumerate(values):
            results.append(oef.filter(v, timestamp=i * 0.033))
        
        # Results should be smoother than input
        input_diff = max(values) - min(values)
        output_diff = max(results) - min(results)
        assert output_diff <= input_diff
    
    def test_responsiveness_fast_signal(self):
        """Test responsiveness to fast changes."""
        oef = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        
        # Simulate fast movement
        oef.filter(0.0, timestamp=0.0)
        result = oef.filter(100.0, timestamp=0.033)
        
        # With high beta, should track fast movements better
        assert result > 50.0  # Should be responsive
    
    def test_reset(self):
        """Test filter reset."""
        oef = OneEuroFilter()
        oef.filter(100.0)
        oef.filter(200.0)
        oef.reset()
        
        # After reset, should act as new filter
        result = oef.filter(50.0)
        assert result == 50.0


class TestKeypointFilter:
    """Tests for KeypointFilter class."""
    
    def test_initialization(self):
        """Test keypoint filter initialization."""
        kf = KeypointFilter(num_keypoints=17, num_dimensions=2)
        assert kf.num_keypoints == 17
        assert kf.num_dimensions == 2
    
    def test_filter_keypoints(self):
        """Test filtering of keypoint array."""
        kf = KeypointFilter(num_keypoints=17, num_dimensions=2)
        
        # Create sample keypoints
        kps = np.random.rand(17, 2).astype(np.float32) * 100
        
        result = kf.filter(kps)
        
        assert result.shape == (17, 2)
        # First frame should pass through relatively unchanged
        np.testing.assert_array_almost_equal(result, kps, decimal=1)
    
    def test_smoothing_over_time(self):
        """Test that filter smooths keypoints over time."""
        kf = KeypointFilter(num_keypoints=17, num_dimensions=2)
        
        # Create base keypoints
        base = np.ones((17, 2), dtype=np.float32) * 100
        
        # First frame
        kf.filter(base, timestamp=0.0)
        
        # Second frame with noise
        noisy = base + np.random.randn(17, 2).astype(np.float32) * 10
        result = kf.filter(noisy, timestamp=0.033)
        
        # Result should be closer to base than noisy input
        base_dist = np.mean(np.abs(result - base))
        noisy_dist = np.mean(np.abs(noisy - base))
        assert base_dist < noisy_dist
    
    def test_reset(self):
        """Test keypoint filter reset."""
        kf = KeypointFilter()
        
        kps1 = np.ones((17, 2), dtype=np.float32) * 100
        kf.filter(kps1)
        kf.reset()
        
        kps2 = np.ones((17, 2), dtype=np.float32) * 50
        result = kf.filter(kps2)
        
        # After reset, should start fresh
        np.testing.assert_array_almost_equal(result, kps2, decimal=1)
    
    def test_update_parameters(self):
        """Test updating filter parameters."""
        kf = KeypointFilter(min_cutoff=1.0, beta=0.007)
        kf.update_parameters(min_cutoff=2.0, beta=0.01)
        
        # Parameters should be updated on internal filters
        assert kf._filters[0][0].min_cutoff == 2.0
        assert kf._filters[0][0].beta == 0.01
