"""Tests for configuration settings."""

import pytest
from pydantic import ValidationError

from squat_analyzer.config.settings import (
    Settings,
    RulesConfig,
    KneeFlexionConfig,
    FilterConfig,
    PoseModelType,
)


class TestSettings:
    """Tests for main Settings class."""
    
    def test_default_settings(self):
        """Test that default settings are valid."""
        settings = Settings()
        assert settings.app_name == "Squat Analyzer"
        assert settings.debug is False
        assert settings.mode.value == "realtime"
    
    def test_pose_config_defaults(self):
        """Test pose estimation defaults."""
        settings = Settings()
        assert settings.pose.model == PoseModelType.YOLOV8N
        assert settings.pose.confidence_threshold == 0.5
        assert settings.pose.device == "auto"
    
    def test_filter_config_defaults(self):
        """Test filter configuration defaults."""
        settings = Settings()
        assert settings.filter.enabled is True
        assert settings.filter.min_cutoff == 1.0
        assert settings.filter.beta == 0.007
    
    def test_rules_config_defaults(self):
        """Test rule configuration defaults."""
        settings = Settings()
        assert settings.rules.knee_flexion.enabled is True
        assert settings.rules.knee_flexion.min_threshold == 70.0
        assert settings.rules.knee_flexion.max_threshold == 135.0
        assert settings.rules.knee_valgus.max_threshold == 10.0


class TestRuleConfig:
    """Tests for individual rule configurations."""
    
    def test_knee_flexion_config(self):
        """Test knee flexion rule config."""
        config = KneeFlexionConfig()
        assert config.severity == 8
        assert config.min_threshold == 70.0
        assert config.max_threshold == 135.0
    
    def test_invalid_threshold_order(self):
        """Test that min > max raises error."""
        from squat_analyzer.config.settings import RuleConfig
        
        with pytest.raises(ValidationError):
            RuleConfig(min_threshold=100.0, max_threshold=50.0)


class TestFilterConfig:
    """Tests for filter configuration."""
    
    def test_filter_config_validation(self):
        """Test filter config validation."""
        # Valid config
        config = FilterConfig(min_cutoff=0.5, beta=0.01)
        assert config.min_cutoff == 0.5
        
        # Invalid min_cutoff
        with pytest.raises(ValidationError):
            FilterConfig(min_cutoff=0.0)
