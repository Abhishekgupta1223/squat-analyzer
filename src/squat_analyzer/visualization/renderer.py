"""
Real-Time Visualization Overlay Renderer
========================================

Provides comprehensive visual feedback including:
    - Skeleton visualization with color-coded joints
    - Real-time angle annotations
    - Feedback message display
    - Performance metrics overlay
    - Rep counter and phase indicator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from squat_analyzer.core.keypoints import (
    Keypoints,
    KeypointIndex,
    SKELETON_CONNECTIONS,
)
from squat_analyzer.analysis.biomechanics import RuleResult, RuleStatus
from squat_analyzer.analysis import SquatPhase
from squat_analyzer.analysis.feedback import FeedbackMessage, FeedbackPriority
from squat_analyzer.utils.logging import get_logger

if TYPE_CHECKING:
    from squat_analyzer.config.settings import VisualizationConfig

logger = get_logger(__name__)


# Color constants (BGR format for OpenCV)
class Colors:
    """Color palette for visualization."""
    
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    GRAY = (128, 128, 128)
    DARK_GRAY = (64, 64, 64)
    
    # Semantic colors
    PASS = GREEN
    WARNING = ORANGE
    FAIL = RED
    INFO = CYAN


class OverlayRenderer:
    """
    Renders visual overlays on video frames.
    
    Provides skeleton visualization, angle annotations, feedback
    display, and metrics overlay for real-time squat analysis.
    
    Example:
        >>> renderer = OverlayRenderer()
        >>> frame = renderer.render(
        ...     frame=video_frame,
        ...     keypoints=detected_keypoints,
        ...     results=analysis_results,
        ...     phase=current_phase,
        ...     rep_count=5,
        ... )
        >>> cv2.imshow("Analysis", frame)
    """
    
    def __init__(
        self,
        config: Optional["VisualizationConfig"] = None,
    ) -> None:
        """
        Initialize the overlay renderer.
        
        Args:
            config: Visualization configuration
        """
        # Store config or use defaults
        self._show_skeleton = True
        self._show_keypoints = True
        self._show_angles = True
        self._show_feedback = True
        self._show_metrics = True
        self._show_phase = True
        
        self._skeleton_color = Colors.GREEN
        self._keypoint_color = Colors.BLUE
        self._line_thickness = 2
        self._keypoint_radius = 5
        self._font_scale = 0.6
        
        if config:
            self._show_skeleton = config.show_skeleton
            self._show_keypoints = config.show_keypoints
            self._show_angles = config.show_angles
            self._show_feedback = config.show_feedback
            self._show_metrics = config.show_metrics
            self._show_phase = config.show_phase
            self._skeleton_color = tuple(config.skeleton_color)
            self._keypoint_color = tuple(config.keypoint_color)
            self._line_thickness = config.line_thickness
            self._keypoint_radius = config.keypoint_radius
            self._font_scale = config.font_scale
        
        # Font settings
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        
        logger.info("OverlayRenderer initialized")
    
    def render(
        self,
        frame: NDArray[np.uint8],
        keypoints: Optional[Keypoints] = None,
        results: Optional[list[RuleResult]] = None,
        feedback: Optional[list[FeedbackMessage]] = None,
        phase: Optional[SquatPhase] = None,
        rep_count: int = 0,
        fps: float = 0.0,
        angles: Optional[dict[str, float]] = None,
        message: Optional[str] = None,
    ) -> NDArray[np.uint8]:
        """
        Render all overlays onto the frame.
        
        Args:
            frame: Input video frame (BGR)
            keypoints: Detected body keypoints
            results: Biomechanical analysis results
            feedback: Feedback messages to display
            phase: Current squat phase
            rep_count: Current rep count
            fps: Current frames per second
            angles: Computed angles for annotation
            message: Override message to display
            
        Returns:
            Frame with overlays rendered
        """
        # Work on a copy
        output = frame.copy()
        h, w = output.shape[:2]
        
        # If there's an override message (like "stand back"), show it prominently
        if message:
            self._draw_message_overlay(output, message)
            if keypoints is not None and self._show_skeleton:
                output = self._draw_skeleton(output, keypoints, None)
            if self._show_metrics:
                output = self._draw_metrics(output, fps, rep_count)
            return output
        
        # Render skeleton and keypoints
        if keypoints is not None:
            if self._show_skeleton:
                output = self._draw_skeleton(output, keypoints, results)
            if self._show_keypoints:
                output = self._draw_keypoints(output, keypoints)
            if self._show_angles and angles:
                output = self._draw_angles(output, keypoints, angles)
        
        # Render feedback panel
        if self._show_feedback:
            output = self._draw_feedback_panel(
                output, feedback, results
            )
        
        # Render metrics and status
        if self._show_metrics:
            output = self._draw_metrics(output, fps, rep_count)
        
        if self._show_phase and phase:
            output = self._draw_phase(output, phase)
        
        return output
    
    def _draw_message_overlay(
        self,
        frame: NDArray[np.uint8],
        message: str,
    ) -> None:
        """Draw a prominent message overlay."""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 80), Colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw message text
        cv2.putText(
            frame, message,
            (20, 55),
            self._font, 1.0, Colors.YELLOW, 2, cv2.LINE_AA
        )
    
    def _draw_skeleton(
        self,
        frame: NDArray[np.uint8],
        keypoints: Keypoints,
        results: Optional[list[RuleResult]] = None,
    ) -> NDArray[np.uint8]:
        """Draw skeleton connections with bounds checking."""
        h, w = frame.shape[:2]
        
        # Determine colors based on results
        base_color = self._skeleton_color
        if results:
            violations = [r for r in results if not r.is_passing]
            if any(r.is_critical for r in violations):
                base_color = Colors.FAIL
            elif violations:
                base_color = Colors.WARNING
        
        for i, j in SKELETON_CONNECTIONS:
            try:
                pt1_raw = keypoints[i]
                pt2_raw = keypoints[j]
                
                # Skip if NaN or invalid
                if np.any(np.isnan(pt1_raw)) or np.any(np.isnan(pt2_raw)):
                    continue
                
                pt1 = (int(np.clip(pt1_raw[0], 0, w-1)), int(np.clip(pt1_raw[1], 0, h-1)))
                pt2 = (int(np.clip(pt2_raw[0], 0, w-1)), int(np.clip(pt2_raw[1], 0, h-1)))
                
                # Skip if either point is at origin (not detected)
                if pt1[0] <= 0 or pt2[0] <= 0:
                    continue
                
                # Check confidence for these joints
                conf1 = keypoints.get_confidence(i)
                conf2 = keypoints.get_confidence(j)
                
                if conf1 < 0.3 or conf2 < 0.3:
                    cv2.line(frame, pt1, pt2, Colors.GRAY, 1)
                else:
                    cv2.line(frame, pt1, pt2, base_color, self._line_thickness)
            except Exception:
                continue  # Skip problematic connections
        
        return frame
    
    def _draw_keypoints(
        self,
        frame: NDArray[np.uint8],
        keypoints: Keypoints,
    ) -> NDArray[np.uint8]:
        """Draw keypoint markers with bounds checking."""
        h, w = frame.shape[:2]
        
        for i in range(17):
            try:
                pt = keypoints[i]
                
                # Skip invalid points
                if np.any(np.isnan(pt)) or pt[0] <= 0:
                    continue
                
                # Clamp to frame bounds
                center = (int(np.clip(pt[0], 0, w-1)), int(np.clip(pt[1], 0, h-1)))
                conf = keypoints.get_confidence(i)
                
                if conf >= 0.5:
                    cv2.circle(frame, center, self._keypoint_radius, self._keypoint_color, -1)
                    cv2.circle(frame, center, self._keypoint_radius, Colors.WHITE, 1)
                elif conf >= 0.3:
                    cv2.circle(frame, center, self._keypoint_radius, Colors.GRAY, 1)
            except Exception:
                continue  # Skip problematic keypoints
        
        return frame
    
    def _draw_angles(
        self,
        frame: NDArray[np.uint8],
        keypoints: Keypoints,
        angles: dict[str, float],
    ) -> NDArray[np.uint8]:
        """Draw angle annotations at key joints with bounds checking."""
        h, w = frame.shape[:2]
        
        def safe_position(pt, offset_x=20, offset_y=0):
            """Get safe drawing position within frame bounds."""
            x = int(np.clip(pt[0] + offset_x, 10, w - 100))
            y = int(np.clip(pt[1] + offset_y, 20, h - 10))
            return (x, y)
        
        try:
            # Knee angle annotation
            if "knee_flexion" in angles and not np.isnan(angles['knee_flexion']):
                knee_mid = (keypoints.left_knee + keypoints.right_knee) / 2
                if not np.any(np.isnan(knee_mid)):
                    self._draw_angle_label(
                        frame,
                        safe_position(knee_mid),
                        f"Knee: {angles['knee_flexion']:.0f}",
                    )
            
            # Hip angle annotation
            if "hip_flexion" in angles and not np.isnan(angles['hip_flexion']):
                hip_mid = keypoints.mid_hip
                if not np.any(np.isnan(hip_mid)):
                    self._draw_angle_label(
                        frame,
                        safe_position(hip_mid),
                        f"Hip: {angles['hip_flexion']:.0f}",
                    )
            
            # Torso angle annotation
            if "torso_inclination" in angles and not np.isnan(angles['torso_inclination']):
                shoulder_mid = keypoints.mid_shoulder
                if not np.any(np.isnan(shoulder_mid)):
                    self._draw_angle_label(
                        frame,
                        safe_position(shoulder_mid),
                        f"Torso: {angles['torso_inclination']:.0f}",
                    )
        except Exception:
            pass  # Skip angle labels on error
        
        return frame
    
    def _draw_angle_label(
        self,
        frame: NDArray[np.uint8],
        position: tuple[int, int],
        text: str,
        color: tuple[int, int, int] = Colors.CYAN,
    ) -> None:
        """Draw an angle label with background."""
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, self._font, self._font_scale * 0.8, 1
        )
        
        # Draw background
        x, y = position
        cv2.rectangle(
            frame,
            (x - 2, y - text_h - 2),
            (x + text_w + 2, y + 2),
            Colors.DARK_GRAY,
            -1,
        )
        
        # Draw text
        cv2.putText(
            frame, text, (x, y),
            self._font, self._font_scale * 0.8, color, 1, cv2.LINE_AA
        )
    
    def _draw_feedback_panel(
        self,
        frame: NDArray[np.uint8],
        feedback: Optional[list[FeedbackMessage]],
        results: Optional[list[RuleResult]],
    ) -> NDArray[np.uint8]:
        """Draw feedback messages panel."""
        h, w = frame.shape[:2]
        
        # Panel settings
        panel_x = 10
        panel_y = 10
        panel_width = 350
        line_height = 25
        
        messages_to_show: list[tuple[str, tuple[int, int, int]]] = []
        
        # Add feedback messages
        if feedback:
            for msg in feedback[:3]:
                color = self._priority_to_color(msg.priority)
                messages_to_show.append((msg.text, color))
                if msg.correction:
                    messages_to_show.append((f"  -> {msg.correction}", color))
        
        # Add rule violations if no feedback
        elif results:
            violations = [r for r in results if not r.is_passing]
            violations.sort(key=lambda r: r.severity, reverse=True)
            for v in violations[:3]:
                color = Colors.FAIL if v.status == RuleStatus.FAIL else Colors.WARNING
                messages_to_show.append((v.message, color))
        
        # Only show "Form looks good" if we actually analyzed something
        if not messages_to_show:
            if results and len(results) > 0:
                # We analyzed and found no issues - form is good!
                messages_to_show.append(("Form looks good!", Colors.PASS))
            else:
                # No analysis performed - don't claim form is good
                messages_to_show.append(("Waiting for analysis...", Colors.GRAY))
        
        # Draw panel background
        panel_height = len(messages_to_show) * line_height + 20
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            Colors.BLACK,
            -1,
        )
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw messages
        for i, (text, color) in enumerate(messages_to_show):
            y = panel_y + 20 + i * line_height
            cv2.putText(
                frame, text[:45],  # Truncate long messages
                (panel_x + 10, y),
                self._font, self._font_scale, color, 1, cv2.LINE_AA
            )
        
        return frame
    
    def _draw_metrics(
        self,
        frame: NDArray[np.uint8],
        fps: float,
        rep_count: int,
    ) -> NDArray[np.uint8]:
        """Draw performance metrics and rep counter."""
        h, w = frame.shape[:2]
        
        # Rep counter (large, top right)
        rep_text = f"REPS: {rep_count}"
        (text_w, text_h), _ = cv2.getTextSize(rep_text, self._font, 1.2, 2)
        
        # Background
        cv2.rectangle(
            frame,
            (w - text_w - 30, 10),
            (w - 10, 10 + text_h + 20),
            Colors.DARK_GRAY,
            -1,
        )
        
        cv2.putText(
            frame, rep_text,
            (w - text_w - 20, 10 + text_h + 10),
            self._font, 1.2, Colors.WHITE, 2, cv2.LINE_AA
        )
        
        # FPS counter (bottom left)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text,
            (10, h - 10),
            self._font, self._font_scale, Colors.GRAY, 1, cv2.LINE_AA
        )
        
        return frame
    
    def _draw_phase(
        self,
        frame: NDArray[np.uint8],
        phase: SquatPhase,
    ) -> NDArray[np.uint8]:
        """Draw current squat phase indicator."""
        h, w = frame.shape[:2]
        
        phase_text = f"Phase: {phase}"
        
        # Color based on phase
        phase_colors = {
            SquatPhase.STANDING: Colors.GRAY,
            SquatPhase.DESCENDING: Colors.CYAN,
            SquatPhase.BOTTOM: Colors.YELLOW,
            SquatPhase.ASCENDING: Colors.GREEN,
        }
        color = phase_colors.get(phase, Colors.WHITE)
        
        cv2.putText(
            frame, phase_text,
            (w - 200, h - 10),
            self._font, self._font_scale, color, 1, cv2.LINE_AA
        )
        
        return frame
    
    def _priority_to_color(
        self,
        priority: FeedbackPriority,
    ) -> tuple[int, int, int]:
        """Convert feedback priority to display color."""
        return {
            FeedbackPriority.CRITICAL: Colors.FAIL,
            FeedbackPriority.HIGH: Colors.FAIL,
            FeedbackPriority.MEDIUM: Colors.WARNING,
            FeedbackPriority.LOW: Colors.CYAN,
            FeedbackPriority.INFO: Colors.WHITE,
        }.get(priority, Colors.WHITE)
