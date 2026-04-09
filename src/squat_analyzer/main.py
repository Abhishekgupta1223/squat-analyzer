"""
Main Application Orchestrator
============================

The central entry point that orchestrates all components:
    - Video/camera capture
    - Pose estimation
    - Signal filtering
    - Biomechanical analysis
    - Squat phase detection
    - Feedback generation
    - Visualization

Supports multiple modes:
    - Real-time webcam analysis
    - Video file processing
    - Single image analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from squat_analyzer.config.settings import Settings, AnalysisMode
from squat_analyzer.core.pose_estimator import PoseEstimator
from squat_analyzer.core.keypoints import Keypoints
from squat_analyzer.core.angles import AngleCalculator
from squat_analyzer.filtering.one_euro import KeypointFilter
from squat_analyzer.analysis.biomechanics import BiomechanicsEngine
from squat_analyzer.analysis.squat_detector import SquatDetector
from squat_analyzer.analysis.feedback import FeedbackGenerator
from squat_analyzer.visualization.renderer import OverlayRenderer
from squat_analyzer.utils.logging import setup_logging, get_logger
from squat_analyzer.utils.metrics import PerformanceMetrics

logger = get_logger(__name__)


class SquatAnalyzer:
    """
    Main application class for squat posture analysis.
    
    Orchestrates all components to provide real-time feedback
    on squat form and technique.
    
    Example:
        >>> from squat_analyzer import SquatAnalyzer, Settings
        >>> settings = Settings()
        >>> analyzer = SquatAnalyzer(settings)
        >>> analyzer.run()  # Start webcam analysis
        
        >>> # Or analyze a video file
        >>> analyzer.run(source="workout.mp4")
    """
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize the squat analyzer.
        
        Args:
            settings: Application settings. Uses defaults if None.
        """
        self._settings = settings or Settings()
        
        # Setup logging
        setup_logging(
            level=self._settings.log_level.value,
            json_output=not self._settings.debug,
        )
        
        # Initialize components
        logger.info("Initializing SquatAnalyzer...")
        
        self._pose_estimator = PoseEstimator(config=self._settings.pose)
        self._angle_calculator = AngleCalculator()
        self._biomechanics = BiomechanicsEngine(config=self._settings.rules)
        self._squat_detector = SquatDetector()
        self._feedback_generator = FeedbackGenerator()
        self._renderer = OverlayRenderer(config=self._settings.visualization)
        self._metrics = PerformanceMetrics()
        
        # Initialize filter if enabled
        self._keypoint_filter: Optional[KeypointFilter] = None
        if self._settings.filter.enabled:
            self._keypoint_filter = KeypointFilter(
                num_keypoints=17,
                num_dimensions=2,
                min_cutoff=self._settings.filter.min_cutoff,
                beta=self._settings.filter.beta,
                d_cutoff=self._settings.filter.d_cutoff,
            )
        
        # Runtime state
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        
        logger.info("SquatAnalyzer initialized successfully")
    
    def run(
        self,
        source: Optional[Union[int, str, Path]] = None,
    ) -> None:
        """
        Run the squat analyzer.
        
        Args:
            source: Video source. Can be:
                    - None: Use settings default (usually webcam 0)
                    - int: Webcam index
                    - str/Path: Video file path or RTSP URL
        """
        # Resolve source
        if source is None:
            source = self._settings.camera.source
        
        # Open video capture
        logger.info("Opening video source", source=str(source))
        self._cap = cv2.VideoCapture(source)
        
        if not self._cap.isOpened():
            logger.error("Failed to open video source", source=str(source))
            raise RuntimeError(f"Could not open video source: {source}")
        
        # Configure capture
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.camera.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.camera.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._settings.camera.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._settings.camera.buffer_size)
        
        # Warm up the model
        logger.info("Warming up pose model...")
        self._pose_estimator.warmup()
        
        # Main loop
        logger.info("Starting analysis loop. Press 'q' to quit.")
        self._running = True
        self._metrics.reset()
        
        try:
            while self._running:
                self._metrics.start_frame()
                
                # Read frame
                ret, frame = self._cap.read()
                if not ret:
                    # End of video file
                    if isinstance(source, (str, Path)):
                        logger.info("End of video file")
                        break
                    # Camera error - try to recover
                    logger.warning("Failed to read frame, attempting recovery...")
                    self._cap.release()
                    self._cap = cv2.VideoCapture(source)
                    continue
                
                # Process frame with error handling
                try:
                    output_frame = self._process_frame(frame)
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    output_frame = frame  # Show raw frame on error
                
                self._metrics.end_frame()
                
                # Display with error handling
                try:
                    cv2.imshow("Squat Analyzer", output_frame)
                except Exception as e:
                    logger.error(f"Display error: {e}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('r'):
                    self._reset()
                elif key == ord('s'):
                    self._save_snapshot(output_frame)
                
                # Periodic cleanup to prevent memory buildup
                if self._metrics.total_frames % 100 == 0:
                    import gc
                    gc.collect()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup()
    
    def _process_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Process a single frame through the analysis pipeline.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Frame with visualization overlays
        """
        # Pose estimation
        keypoints = self._pose_estimator.estimate_single(frame)
        
        if keypoints is None:
            # No person detected
            return self._renderer.render(
                frame=frame,
                fps=self._metrics.avg_fps,
                message="No person detected",
            )
        
        # CHECK: Are the LEGS actually visible? 
        # EXTREMELY lenient - only block if upper body is crystal clear but legs are invisible
        # This allows detection at ANY distance where pose model can see the person
        upper_keypoints = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
        upper_confidences = [keypoints.get_confidence(i) for i in upper_keypoints]
        avg_upper = sum(upper_confidences) / len(upper_confidences)
        
        leg_keypoints = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
        leg_confidences = [keypoints.get_confidence(i) for i in leg_keypoints]
        avg_leg = sum(leg_confidences) / len(leg_confidences)
        
        # Only block if upper body is VERY clear but legs are COMPLETELY invisible
        # This should almost never trigger when full body is visible (even at distance)
        legs_not_visible = (avg_upper > 0.7 and avg_leg < 0.1)
        
        if legs_not_visible:
            return self._renderer.render(
                frame=frame,
                keypoints=keypoints,
                fps=self._metrics.avg_fps,
                message="Step back - show full body",
            )
        
        # Apply filtering
        if self._keypoint_filter is not None:
            filtered_points = self._keypoint_filter.filter(keypoints.points)
            keypoints = Keypoints(points=filtered_points, confidence=keypoints.confidence)
        
        # Calculate angles
        angles = self._angle_calculator.compute_all_angles(keypoints)
        
        # Biomechanical analysis
        results = self._biomechanics.analyze(keypoints)
        
        # Squat phase detection
        phase = self._squat_detector.update(keypoints)
        
        # Generate feedback
        feedback = self._feedback_generator.generate(
            results=results,
            phase=phase,
            rep_count=self._squat_detector.rep_count,
        )
        
        # Render visualization
        output = self._renderer.render(
            frame=frame,
            keypoints=keypoints,
            results=results,
            feedback=feedback,
            phase=phase,
            rep_count=self._squat_detector.rep_count,
            fps=self._metrics.avg_fps,
            angles=angles,
        )
        
        return output
    
    def analyze_image(self, image_path: Union[str, Path]) -> dict:
        """
        Analyze a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Analysis results dictionary
        """
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Estimate pose
        keypoints = self._pose_estimator.estimate_single(frame)
        if keypoints is None:
            return {"error": "No person detected"}
        
        # Calculate angles
        angles = self._angle_calculator.compute_all_angles(keypoints)
        
        # Analyze
        results = self._biomechanics.analyze(keypoints)
        
        # Build response
        return {
            "angles": angles,
            "overall_score": self._biomechanics.overall_score(results),
            "rules": [
                {
                    "name": r.name,
                    "status": r.status.name,
                    "score": r.score,
                    "value": r.value,
                    "message": r.message,
                    "correction": r.correction,
                }
                for r in results
            ],
        }
    
    def _reset(self) -> None:
        """Reset analysis state."""
        logger.info("Resetting analysis state")
        self._squat_detector.reset()
        self._feedback_generator.reset()
        if self._keypoint_filter:
            self._keypoint_filter.reset()
        self._metrics.reset()
    
    def _save_snapshot(self, frame: NDArray[np.uint8]) -> None:
        """Save current frame as image."""
        import time
        filename = f"squat_snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        logger.info("Snapshot saved", filename=filename)
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False
        
        if self._cap is not None:
            self._cap.release()
        
        cv2.destroyAllWindows()
        
        # Print summary
        logger.info(
            "Session complete",
            total_frames=self._metrics.total_frames,
            overall_fps=f"{self._metrics.overall_fps:.1f}",
            total_reps=self._squat_detector.rep_count,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Squat Analyzer - Real-time posture analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--source", "-s",
        default=0,
        help="Video source (webcam index, file path, or RTSP URL)",
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--image", "-i",
        type=Path,
        help="Analyze a single image instead of video",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Load settings
    if args.config and args.config.exists():
        settings = Settings.from_yaml(args.config)
    else:
        settings = Settings()
    
    if args.debug:
        settings.debug = True
        settings.log_level = "DEBUG"
    
    # Create analyzer
    analyzer = SquatAnalyzer(settings)
    
    # Run appropriate mode
    if args.image:
        result = analyzer.analyze_image(args.image)
        import json
        print(json.dumps(result, indent=2))
    else:
        # Parse source
        source = args.source
        try:
            source = int(source)
        except ValueError:
            pass  # Keep as string (file path or URL)
        
        analyzer.run(source=source)


if __name__ == "__main__":
    main()
