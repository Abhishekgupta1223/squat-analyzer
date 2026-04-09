"""Tests for biomechanics analysis engine."""

import pytest

from squat_analyzer.analysis.biomechanics import (
    BiomechanicsEngine,
    RuleResult,
    RuleStatus,
    KneeFlexionRule,
    KneeValgusRule,
    TorsoInclinationRule,
)
from squat_analyzer.core.angles import AngleCalculator


class TestBiomechanicsEngine:
    """Tests for BiomechanicsEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create biomechanics engine instance."""
        return BiomechanicsEngine()
    
    def test_engine_initialization(self, engine):
        """Test engine initializes with all rules."""
        assert len(engine.rules) == 6
    
    def test_analyze_standing(self, engine, sample_keypoints):
        """Test analysis of standing pose."""
        results = engine.analyze(sample_keypoints)
        
        # Should get results for all rules
        assert len(results) == 6
        
        # All results should be RuleResult objects
        for result in results:
            assert isinstance(result, RuleResult)
    
    def test_analyze_squat(self, engine, squat_bottom_keypoints):
        """Test analysis of squat position."""
        results = engine.analyze(squat_bottom_keypoints)
        assert len(results) > 0
    
    def test_overall_score(self, engine, sample_keypoints):
        """Test overall score calculation."""
        results = engine.analyze(sample_keypoints)
        score = engine.overall_score(results)
        
        assert 0 <= score <= 100
    
    def test_get_violations(self, engine, squat_bottom_keypoints):
        """Test getting rule violations."""
        results = engine.analyze(squat_bottom_keypoints)
        violations = engine.get_violations(results)
        
        # Violations should be sorted by severity
        if len(violations) > 1:
            assert violations[0].severity >= violations[-1].severity
    
    def test_empty_results_score(self, engine):
        """Test score with empty results."""
        score = engine.overall_score([])
        assert score == 0.0


class TestKneeFlexionRule:
    """Tests for knee flexion rule."""
    
    @pytest.fixture
    def rule(self):
        """Create knee flexion rule."""
        return KneeFlexionRule(min_angle=70.0, max_angle=135.0)
    
    @pytest.fixture
    def calculator(self):
        """Create angle calculator."""
        return AngleCalculator()
    
    def test_passes_in_range(self, rule, sample_keypoints, calculator):
        """Test rule passes when angle in range."""
        angles = {"knee_flexion": 100.0}
        result = rule.evaluate(sample_keypoints, angles)
        
        assert result.status == RuleStatus.PASS
        assert result.score == 100.0
    
    def test_fails_too_shallow(self, rule, sample_keypoints, calculator):
        """Test rule fails when squat too shallow."""
        angles = {"knee_flexion": 160.0}  # Almost straight
        result = rule.evaluate(sample_keypoints, angles)
        
        assert result.status in [RuleStatus.WARNING, RuleStatus.FAIL]
        assert result.score < 100.0
    
    def test_warning_slightly_deep(self, rule, sample_keypoints, calculator):
        """Test rule warns when slightly too deep."""
        angles = {"knee_flexion": 60.0}  # Slightly below min
        result = rule.evaluate(sample_keypoints, angles)
        
        assert result.status == RuleStatus.WARNING


class TestKneeValgusRule:
    """Tests for knee valgus rule."""
    
    @pytest.fixture
    def rule(self):
        """Create knee valgus rule."""
        return KneeValgusRule(max_valgus=10.0)
    
    def test_passes_no_valgus(self, rule, sample_keypoints):
        """Test rule passes with no valgus."""
        angles = {"knee_valgus": 5.0}
        result = rule.evaluate(sample_keypoints, angles)
        
        assert result.status == RuleStatus.PASS
    
    def test_fails_excessive_valgus(self, rule, sample_keypoints):
        """Test rule fails with excessive valgus."""
        angles = {"knee_valgus": 20.0}
        result = rule.evaluate(sample_keypoints, angles)
        
        assert result.status == RuleStatus.FAIL
        assert "inward" in result.message.lower()


class TestTorsoInclinationRule:
    """Tests for torso inclination rule."""
    
    @pytest.fixture
    def rule(self):
        """Create torso rule."""
        return TorsoInclinationRule(min_lean=30.0, max_lean=75.0)
    
    def test_passes_good_lean(self, rule, sample_keypoints):
        """Test rule passes with good lean angle."""
        angles = {"torso_inclination": 45.0}
        result = rule.evaluate(sample_keypoints, angles)
        
        assert result.status == RuleStatus.PASS
    
    def test_fails_excessive_lean(self, rule, sample_keypoints):
        """Test rule fails with excessive forward lean."""
        angles = {"torso_inclination": 85.0}
        result = rule.evaluate(sample_keypoints, angles)
        
        assert result.status in [RuleStatus.WARNING, RuleStatus.FAIL]
        assert "lean" in result.message.lower() or "chest" in result.correction.lower()
