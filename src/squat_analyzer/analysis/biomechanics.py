"""
Research-Grade Biomechanics Analysis Engine
===========================================

Implements 6 evidence-based biomechanical rules for squat assessment
based on peer-reviewed sports science literature.

Research Foundation:
    1. Knee Flexion: Schoenfeld (2010) - JSCR
    2. Knee Valgus: Hewett et al. (2005) - AJSM  
    3. Torso Inclination: Escamilla (2001) - MSSE
    4. Hip Hinge: Hartmann et al. (2013) - Sports Medicine
    5. Knee-Over-Toe: Fry et al. (2003) - JSCR
    6. Depth Analysis: NSCA Guidelines

Each rule provides:
    - Binary pass/fail assessment
    - Continuous score [0-100]
    - Specific corrective feedback
    - Severity rating for prioritization
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np

from squat_analyzer.core.angles import AngleCalculator
from squat_analyzer.core.keypoints import Keypoints
from squat_analyzer.utils.logging import get_logger

if TYPE_CHECKING:
    from squat_analyzer.config.settings import RulesConfig

logger = get_logger(__name__)


class RuleStatus(Enum):
    """Status of a biomechanical rule check."""
    
    PASS = auto()
    WARNING = auto()
    FAIL = auto()
    NOT_APPLICABLE = auto()


@dataclass(frozen=True)
class RuleResult:
    """
    Result of evaluating a biomechanical rule.
    
    Attributes:
        name: Human-readable rule name
        status: Pass/Warning/Fail status
        score: Continuous score [0-100], 100 = perfect
        value: Measured value (e.g., angle in degrees)
        threshold_min: Minimum acceptable value
        threshold_max: Maximum acceptable value
        message: Descriptive feedback message
        severity: Priority level [1-10], 10 = most critical
        correction: Specific corrective instruction
    """
    
    name: str
    status: RuleStatus
    score: float  # 0-100
    value: float
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    message: str = ""
    severity: int = 5
    correction: str = ""
    
    @property
    def is_passing(self) -> bool:
        """Check if rule is passing."""
        return self.status == RuleStatus.PASS
    
    @property
    def is_critical(self) -> bool:
        """Check if rule failure is critical (severity >= 8)."""
        return self.status == RuleStatus.FAIL and self.severity >= 8


class BiomechanicalRule(ABC):
    """Abstract base class for biomechanical rules."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable rule name."""
        pass
    
    @property
    @abstractmethod
    def severity(self) -> int:
        """Severity level [1-10]."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        keypoints: Keypoints,
        angles: dict[str, float],
    ) -> RuleResult:
        """
        Evaluate the rule against current pose.
        
        Args:
            keypoints: Detected body keypoints
            angles: Pre-computed angles from AngleCalculator
            
        Returns:
            RuleResult with assessment details
        """
        pass


class KneeFlexionRule(BiomechanicalRule):
    """
    Knee Flexion Angle Rule
    
    Monitors knee angle throughout squat movement.
    
    Research: Schoenfeld (2010) showed optimal muscle activation
    at knee angles between 70° and 135°.
    """
    
    def __init__(
        self,
        min_angle: float = 70.0,
        max_angle: float = 135.0,
        severity: int = 8,
    ) -> None:
        self.min_angle = min_angle
        self.max_angle = max_angle
        self._severity = severity
    
    @property
    def name(self) -> str:
        return "Knee Flexion"
    
    @property
    def severity(self) -> int:
        return self._severity
    
    def evaluate(
        self,
        keypoints: Keypoints,
        angles: dict[str, float],
    ) -> RuleResult:
        knee_angle = angles.get("knee_flexion", 180.0)
        
        # Score calculation: 100 at optimal, decreasing linearly outside range
        if self.min_angle <= knee_angle <= self.max_angle:
            # Within optimal range
            score = 100.0
            status = RuleStatus.PASS
            message = f"Excellent depth! Knee angle {knee_angle:.0f}° is optimal"
            correction = ""
        elif knee_angle < self.min_angle:
            # Too deep (less common issue)
            deficit = self.min_angle - knee_angle
            score = max(0, 100 - deficit * 2)
            status = RuleStatus.WARNING if deficit < 15 else RuleStatus.FAIL
            message = f"Very deep squat ({knee_angle:.0f}°) - check mobility"
            correction = "Control descent. If pain occurs, reduce depth slightly"
        else:
            # Not deep enough
            deficit = knee_angle - self.max_angle
            score = max(0, 100 - deficit * 2)
            status = RuleStatus.WARNING if deficit < 20 else RuleStatus.FAIL
            if deficit < 20:
                message = "Go a bit deeper for full muscle activation"
                correction = "Break parallel: hips below knee level"
            else:
                message = "Squat depth insufficient - hips too high"
                correction = "Sit back and down. Imagine sitting into a low chair"
        
        return RuleResult(
            name=self.name,
            status=status,
            score=score,
            value=knee_angle,
            threshold_min=self.min_angle,
            threshold_max=self.max_angle,
            message=message,
            severity=self._severity,
            correction=correction,
        )


class KneeValgusRule(BiomechanicalRule):
    """
    Knee Valgus Detection Rule
    
    Detects inward knee collapse which increases ACL injury risk.
    
    Research: Hewett et al. (2005) identified valgus angles >10°
    as significantly increasing injury risk in athletes.
    """
    
    def __init__(
        self,
        max_valgus: float = 10.0,
        severity: int = 9,
    ) -> None:
        self.max_valgus = max_valgus
        self._severity = severity
    
    @property
    def name(self) -> str:
        return "Knee Valgus"
    
    @property
    def severity(self) -> int:
        return self._severity
    
    def evaluate(
        self,
        keypoints: Keypoints,
        angles: dict[str, float],
    ) -> RuleResult:
        valgus = angles.get("knee_valgus", 0.0)
        
        if valgus <= self.max_valgus:
            score = 100.0 - (valgus / self.max_valgus) * 20
            status = RuleStatus.PASS
            message = "Knees tracking well over toes"
            correction = ""
        else:
            excess = valgus - self.max_valgus
            score = max(0, 80 - excess * 5)
            status = RuleStatus.WARNING if excess < 5 else RuleStatus.FAIL
            if excess < 5:
                message = "Slight knee cave detected - stay aware"
                correction = "Cue: 'Screw feet into floor' - external rotation"
            else:
                message = "⚠️ KNEE VALGUS - Injury risk! Knees collapsing inward"
                correction = "STOP if painful. Push knees OUT over pinky toes"
        
        return RuleResult(
            name=self.name,
            status=status,
            score=score,
            value=valgus,
            threshold_max=self.max_valgus,
            message=message,
            severity=self._severity,
            correction=correction,
        )


class TorsoInclinationRule(BiomechanicalRule):
    """
    Torso Inclination Rule
    
    Monitors forward lean to protect the lower back.
    
    Research: Escamilla (2001) recommends torso inclination
    between 30° and 75° for safe loaded squatting.
    """
    
    def __init__(
        self,
        min_lean: float = 30.0,
        max_lean: float = 75.0,
        severity: int = 7,
    ) -> None:
        self.min_lean = min_lean
        self.max_lean = max_lean
        self._severity = severity
    
    @property
    def name(self) -> str:
        return "Torso Inclination"
    
    @property
    def severity(self) -> int:
        return self._severity
    
    def evaluate(
        self,
        keypoints: Keypoints,
        angles: dict[str, float],
    ) -> RuleResult:
        torso_angle = angles.get("torso_inclination", 0.0)
        
        if self.min_lean <= torso_angle <= self.max_lean:
            score = 100.0
            status = RuleStatus.PASS
            message = "Good torso position - spine neutral"
            correction = ""
        elif torso_angle < self.min_lean:
            # Too upright (uncommon)
            deficit = self.min_lean - torso_angle
            score = max(0, 100 - deficit * 2)
            status = RuleStatus.WARNING
            message = "Torso very upright - allow natural hip hinge"
            correction = "Slight forward lean is normal and safe"
        else:
            # Too much forward lean
            excess = torso_angle - self.max_lean
            score = max(0, 100 - excess * 3)
            status = RuleStatus.WARNING if excess < 15 else RuleStatus.FAIL
            if excess < 15:
                message = "Leaning forward - engage your core"
                correction = "Cue: 'Chest proud' - lift sternum toward ceiling"
            else:
                message = "⚠️ Excessive forward lean - lower back at risk"
                correction = "Brace core HARD. Look forward, not down"
        
        return RuleResult(
            name=self.name,
            status=status,
            score=score,
            value=torso_angle,
            threshold_min=self.min_lean,
            threshold_max=self.max_lean,
            message=message,
            severity=self._severity,
            correction=correction,
        )


class HipHingeRule(BiomechanicalRule):
    """
    Hip Hinge Mechanics Rule
    
    Ensures proper hip flexion for posterior chain engagement.
    
    Research: Hartmann et al. (2013) showed hip angles 45°-100°
    provide optimal recruitment of gluteal muscles.
    """
    
    def __init__(
        self,
        min_angle: float = 45.0,
        max_angle: float = 100.0,
        severity: int = 6,
    ) -> None:
        self.min_angle = min_angle
        self.max_angle = max_angle
        self._severity = severity
    
    @property
    def name(self) -> str:
        return "Hip Hinge"
    
    @property
    def severity(self) -> int:
        return self._severity
    
    def evaluate(
        self,
        keypoints: Keypoints,
        angles: dict[str, float],
    ) -> RuleResult:
        hip_angle = angles.get("hip_flexion", 180.0)
        
        if self.min_angle <= hip_angle <= self.max_angle:
            score = 100.0
            status = RuleStatus.PASS
            message = "Good hip mechanics - glutes engaged"
            correction = ""
        elif hip_angle < self.min_angle:
            deficit = self.min_angle - hip_angle
            score = max(0, 100 - deficit * 2)
            status = RuleStatus.WARNING if deficit < 20 else RuleStatus.FAIL
            message = "Hips not hinging enough - missing glute activation"
            correction = "Push hips BACK first, then down. Feel glutes stretch"
        else:
            excess = hip_angle - self.max_angle
            score = max(0, 100 - excess * 2)
            status = RuleStatus.WARNING
            message = "Limited hip flexion - work on mobility"
            correction = "Hip flexor stretch recommended. Practice box squats"
        
        return RuleResult(
            name=self.name,
            status=status,
            score=score,
            value=hip_angle,
            threshold_min=self.min_angle,
            threshold_max=self.max_angle,
            message=message,
            severity=self._severity,
            correction=correction,
        )


class KneeOverToeRule(BiomechanicalRule):
    """
    Knee-Over-Toe Positioning Rule
    
    Monitors knee position relative to toes.
    
    Research: Fry et al. (2003) found excessive forward knee
    travel increases patellofemoral stress.
    
    Note: Modern research shows some forward travel is acceptable
    and often necessary for proper squat mechanics.
    """
    
    def __init__(
        self,
        max_ratio: float = 0.15,  # 15% past toe
        severity: int = 5,
    ) -> None:
        self.max_ratio = max_ratio
        self._severity = severity
    
    @property
    def name(self) -> str:
        return "Knee Position"
    
    @property
    def severity(self) -> int:
        return self._severity
    
    def evaluate(
        self,
        keypoints: Keypoints,
        angles: dict[str, float],
    ) -> RuleResult:
        ratio = angles.get("knee_over_toe", 0.0)
        
        if ratio <= self.max_ratio:
            score = 100.0 - abs(ratio / self.max_ratio) * 20
            status = RuleStatus.PASS
            message = "Knee position balanced over feet"
            correction = ""
        else:
            excess = ratio - self.max_ratio
            score = max(0, 80 - excess * 200)
            status = RuleStatus.WARNING if excess < 0.1 else RuleStatus.FAIL
            if excess < 0.1:
                message = "Knees slightly forward - shift weight back"
                correction = "Cue: 'Weight in heels' - wiggle your toes"
            else:
                message = "Knees too far forward - heel coming up?"
                correction = "Sit BACK into squat. Heels stay glued to floor"
        
        return RuleResult(
            name=self.name,
            status=status,
            score=score,
            value=ratio * 100,  # Convert to percentage for display
            threshold_max=self.max_ratio * 100,
            message=message,
            severity=self._severity,
            correction=correction,
        )


class DepthAnalysisRule(BiomechanicalRule):
    """
    Squat Depth Analysis Rule
    
    Verifies proper squat depth for full range of motion benefits.
    
    Research: NSCA guidelines recommend thigh parallel or below
    for optimal strength and hypertrophy adaptations.
    """
    
    def __init__(
        self,
        max_above_parallel: float = 15.0,  # degrees above parallel
        severity: int = 5,
    ) -> None:
        self.max_above_parallel = max_above_parallel
        self._severity = severity
    
    @property
    def name(self) -> str:
        return "Depth Analysis"
    
    @property
    def severity(self) -> int:
        return self._severity
    
    def evaluate(
        self,
        keypoints: Keypoints,
        angles: dict[str, float],
    ) -> RuleResult:
        thigh_angle = angles.get("thigh_angle", 90.0)
        
        # thigh_angle: 0 = parallel, negative = below parallel (ATG)
        if thigh_angle <= 0:
            # At or below parallel - excellent
            score = 100.0
            status = RuleStatus.PASS
            message = "Excellent depth - at or below parallel"
            correction = ""
        elif thigh_angle <= self.max_above_parallel:
            # Slightly above parallel - acceptable
            score = 85.0 - thigh_angle
            status = RuleStatus.WARNING
            message = f"Depth acceptable ({thigh_angle:.1f}° above parallel)"
            correction = "Try to go slightly deeper for full ROM"
        else:
            # Not deep enough
            score = max(0, 70 - thigh_angle)
            status = RuleStatus.FAIL
            message = f"Insufficient depth ({thigh_angle:.1f}° above parallel)"
            correction = "Squat lower - aim for thighs parallel to ground"
        
        return RuleResult(
            name=self.name,
            status=status,
            score=score,
            value=thigh_angle,
            threshold_max=self.max_above_parallel,
            message=message,
            severity=self._severity,
            correction=correction,
        )


class BiomechanicsEngine:
    """
    Main engine for biomechanical squat analysis.
    
    Orchestrates all biomechanical rules and provides
    comprehensive assessment of squat form.
    
    Example:
        >>> from squat_analyzer.config import Settings
        >>> settings = Settings()
        >>> engine = BiomechanicsEngine(settings.rules)
        >>> results = engine.analyze(keypoints)
        >>> print(f"Overall score: {engine.overall_score(results):.1f}")
    """
    
    def __init__(
        self,
        config: Optional["RulesConfig"] = None,
    ) -> None:
        """
        Initialize the biomechanics engine.
        
        Args:
            config: Optional RulesConfig for custom thresholds
        """
        self._angle_calculator = AngleCalculator()
        self._rules: list[BiomechanicalRule] = []
        
        self._initialize_rules(config)
        
        logger.info(
            "BiomechanicsEngine initialized",
            num_rules=len(self._rules),
        )
    
    def _initialize_rules(self, config: Optional["RulesConfig"]) -> None:
        """Initialize all biomechanical rules from config."""
        if config is None:
            # Use default rules
            self._rules = [
                KneeFlexionRule(),
                KneeValgusRule(),
                TorsoInclinationRule(),
                HipHingeRule(),
                KneeOverToeRule(),
                DepthAnalysisRule(),
            ]
            return
        
        # Initialize from config
        if config.knee_flexion.enabled:
            self._rules.append(KneeFlexionRule(
                min_angle=config.knee_flexion.min_threshold or 70.0,
                max_angle=config.knee_flexion.max_threshold or 135.0,
                severity=config.knee_flexion.severity,
            ))
        
        if config.knee_valgus.enabled:
            self._rules.append(KneeValgusRule(
                max_valgus=config.knee_valgus.max_threshold or 10.0,
                severity=config.knee_valgus.severity,
            ))
        
        if config.torso_inclination.enabled:
            self._rules.append(TorsoInclinationRule(
                min_lean=config.torso_inclination.min_threshold or 30.0,
                max_lean=config.torso_inclination.max_threshold or 75.0,
                severity=config.torso_inclination.severity,
            ))
        
        if config.hip_hinge.enabled:
            self._rules.append(HipHingeRule(
                min_angle=config.hip_hinge.min_threshold or 45.0,
                max_angle=config.hip_hinge.max_threshold or 100.0,
                severity=config.hip_hinge.severity,
            ))
        
        if config.knee_over_toe.enabled:
            self._rules.append(KneeOverToeRule(
                max_ratio=config.knee_over_toe.max_threshold or 0.15,
                severity=config.knee_over_toe.severity,
            ))
        
        if config.depth_analysis.enabled:
            self._rules.append(DepthAnalysisRule(
                max_above_parallel=config.depth_analysis.max_threshold or 15.0,
                severity=config.depth_analysis.severity,
            ))
    
    def analyze(self, keypoints: Keypoints) -> list[RuleResult]:
        """
        Perform full biomechanical analysis with error handling.
        
        Args:
            keypoints: Detected body keypoints
            
        Returns:
            List of RuleResult for each enabled rule
        """
        # Check if we have SOME keypoints - be lenient for real-world videos
        # Only require 4 of 8 critical keypoints with very low threshold (0.05)
        # since YouTube videos have varying quality and angles
        if not keypoints.has_partial_keypoints(min_count=4, threshold=0.05):
            logger.debug("Insufficient keypoints for analysis")
            return []
        
        # Compute all angles with error handling
        try:
            angles = self._angle_calculator.compute_all_angles(keypoints)
            
            # Only skip if ALL angles are NaN - allow partial analysis
            valid_angles = {k: v for k, v in angles.items() if not np.isnan(v)}
            if not valid_angles:
                logger.debug("No valid angles computed, skipping analysis")
                return []
            # Fill NaN with defaults for partial data
            for key in angles:
                if np.isnan(angles[key]):
                    angles[key] = 180.0  # Default to straight angle
        except Exception as e:
            logger.debug(f"Angle computation failed: {e}")
            return []
        
        # Evaluate all rules with error handling
        results = []
        for rule in self._rules:
            try:
                result = rule.evaluate(keypoints, angles)
                results.append(result)
            except Exception as e:
                logger.debug(f"Rule evaluation failed: {rule.name} - {e}")
        
        return results
    
    def overall_score(self, results: list[RuleResult]) -> float:
        """
        Calculate weighted overall score.
        
        Higher severity rules have more weight in the final score.
        
        Args:
            results: List of RuleResult from analyze()
            
        Returns:
            Weighted average score [0-100]
        """
        if not results:
            return 0.0
        
        total_weight = sum(r.severity for r in results)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(r.score * r.severity for r in results)
        return weighted_sum / total_weight
    
    def get_violations(self, results: list[RuleResult]) -> list[RuleResult]:
        """
        Get all failing or warning results.
        
        Args:
            results: List of RuleResult from analyze()
            
        Returns:
            List of non-passing results, sorted by severity
        """
        violations = [r for r in results if not r.is_passing]
        return sorted(violations, key=lambda r: r.severity, reverse=True)
    
    def get_critical_violations(self, results: list[RuleResult]) -> list[RuleResult]:
        """
        Get only critical violations (severity >= 8).
        
        Args:
            results: List of RuleResult from analyze()
            
        Returns:
            List of critical violations
        """
        return [r for r in results if r.is_critical]
    
    @property
    def rules(self) -> list[BiomechanicalRule]:
        """Get list of active rules."""
        return self._rules
