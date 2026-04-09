"""
Intelligent Feedback Generation System
======================================

Generates prioritized, actionable feedback messages based on
biomechanical analysis results. Includes:
    - Priority queuing (critical issues first)
    - Cooldown to prevent feedback spam
    - Contextual message generation
    - Audio cue support (text-to-speech ready)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time

from squat_analyzer.analysis.biomechanics import RuleResult, RuleStatus
from squat_analyzer.analysis.squat_detector_v2 import SquatPhase
from squat_analyzer.utils.logging import get_logger

logger = get_logger(__name__)


class FeedbackPriority(Enum):
    """Priority levels for feedback messages."""
    
    CRITICAL = auto()   # Injury risk - immediate attention
    HIGH = auto()       # Form issue - address soon
    MEDIUM = auto()     # Optimization - nice to fix
    LOW = auto()        # Minor - informational
    INFO = auto()       # Status updates
    
    def __lt__(self, other: "FeedbackPriority") -> bool:
        return self.value < other.value
    
    def __le__(self, other: "FeedbackPriority") -> bool:
        return self.value <= other.value
    
    def __gt__(self, other: "FeedbackPriority") -> bool:
        return self.value > other.value
    
    def __ge__(self, other: "FeedbackPriority") -> bool:
        return self.value >= other.value


@dataclass(frozen=True)
class FeedbackMessage:
    """
    A feedback message to display to the user.
    
    Attributes:
        text: Main message text
        priority: Message priority level
        rule_name: Source rule (if from biomechanics)
        correction: Suggested fix
        duration: How long to display (seconds)
    """
    
    text: str
    priority: FeedbackPriority
    rule_name: str = ""
    correction: str = ""
    duration: float = 3.0
    
    @property
    def is_critical(self) -> bool:
        """Check if this is a critical message."""
        return self.priority == FeedbackPriority.CRITICAL
    
    @property
    def display_text(self) -> str:
        """Get full display text including correction if available."""
        if self.correction:
            return f"{self.text}\n{self.correction}"
        return self.text


@dataclass
class FeedbackState:
    """Internal state for feedback management."""
    
    # Message history with timestamps
    last_messages: dict[str, float] = field(default_factory=dict)
    
    # Current active message
    active_message: Optional[FeedbackMessage] = None
    active_message_start: float = 0.0
    
    # Rep completion feedback
    last_rep_count: int = 0


class FeedbackGenerator:
    """
    Generates and manages user feedback during squat analysis.
    
    Features:
        - Priority-based message selection
        - Cooldown per message type to prevent spam
        - Phase-aware contextual messages
        - Rep completion announcements
    
    Example:
        >>> generator = FeedbackGenerator()
        >>> messages = generator.generate(analysis_results, phase)
        >>> for msg in messages:
        ...     display(msg.text)
    """
    
    def __init__(
        self,
        message_cooldown: float = 2.0,
        max_displayed_messages: int = 2,
    ) -> None:
        """
        Initialize the feedback generator.
        
        Args:
            message_cooldown: Seconds before same message can repeat
            max_displayed_messages: Maximum concurrent messages
        """
        self._cooldown = message_cooldown
        self._max_messages = max_displayed_messages
        self._state = FeedbackState()
        
        # Phase-specific coaching cues (professional terminology)
        self._phase_messages = {
            SquatPhase.STANDING: "Set position - brace your core",
            SquatPhase.DESCENDING: "Control the descent - breathe in",
            SquatPhase.BOTTOM: "Hold tension - don't relax!",
            SquatPhase.ASCENDING: "Drive through heels - exhale!",
        }
        
        # Rep completion messages (rotating for variety)
        self._rep_celebrations = [
            "Nice rep! Keep that form",
            "Good work! Stay focused",
            "Solid! Maintain technique",
            "Strong rep! Keep it up",
            "Well done! Stay tight",
        ]
        
        logger.info("FeedbackGenerator initialized")
    
    def generate(
        self,
        results: list[RuleResult],
        phase: SquatPhase,
        rep_count: int = 0,
    ) -> list[FeedbackMessage]:
        """
        Generate feedback messages from analysis results.
        
        Args:
            results: Biomechanical analysis results
            phase: Current squat phase
            rep_count: Current rep count
            
        Returns:
            List of feedback messages, sorted by priority
        """
        current_time = time.time()
        messages: list[FeedbackMessage] = []
        
        # Check for rep completion
        if rep_count > self._state.last_rep_count:
            # Use rotating celebration messages
            celebration = self._rep_celebrations[rep_count % len(self._rep_celebrations)]
            messages.append(FeedbackMessage(
                text=f"Rep {rep_count} \u2714 {celebration}",
                priority=FeedbackPriority.INFO,
                duration=1.5,
            ))
            self._state.last_rep_count = rep_count
        
        # Process rule violations
        for result in results:
            if result.status == RuleStatus.PASS:
                continue
            
            # Check cooldown
            msg_key = f"{result.name}_{result.status.name}"
            last_time = self._state.last_messages.get(msg_key, 0)
            if current_time - last_time < self._cooldown:
                continue
            
            # Determine priority
            if result.is_critical:
                priority = FeedbackPriority.CRITICAL
            elif result.status == RuleStatus.FAIL:
                priority = FeedbackPriority.HIGH
            else:
                priority = FeedbackPriority.MEDIUM
            
            # Create message
            message = FeedbackMessage(
                text=result.message,
                priority=priority,
                rule_name=result.name,
                correction=result.correction,
                duration=3.0 if priority <= FeedbackPriority.HIGH else 2.0,
            )
            
            messages.append(message)
            self._state.last_messages[msg_key] = current_time
        
        # Sort by priority and limit
        messages.sort(key=lambda m: m.priority)
        return messages[:self._max_messages]
    
    def get_phase_message(self, phase: SquatPhase) -> str:
        """Get encouragement message for current phase."""
        return self._phase_messages.get(phase, "")
    
    def generate_summary(
        self,
        results: list[RuleResult],
        rep_count: int,
    ) -> FeedbackMessage:
        """
        Generate end-of-set summary feedback.
        
        Args:
            results: Final analysis results
            rep_count: Total reps completed
            
        Returns:
            Summary feedback message
        """
        if not results:
            return FeedbackMessage(
                text=f"Set complete: {rep_count} reps",
                priority=FeedbackPriority.INFO,
            )
        
        # Calculate stats
        passing = sum(1 for r in results if r.is_passing)
        total = len(results)
        avg_score = sum(r.score for r in results) / total if total > 0 else 0
        
        # Find biggest issue
        violations = [r for r in results if not r.is_passing]
        violations.sort(key=lambda r: r.severity, reverse=True)
        
        summary_lines = [f"Set complete: {rep_count} reps"]
        summary_lines.append(f"Form score: {avg_score:.0f}/100")
        
        if violations:
            main_issue = violations[0]
            summary_lines.append(f"Focus area: {main_issue.name}")
        else:
            summary_lines.append("Excellent form!")
        
        return FeedbackMessage(
            text="\n".join(summary_lines),
            priority=FeedbackPriority.INFO,
            duration=5.0,
        )
    
    def reset(self) -> None:
        """Reset feedback state."""
        self._state = FeedbackState()
        logger.info("FeedbackGenerator reset")
