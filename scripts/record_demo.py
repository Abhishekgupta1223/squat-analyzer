"""
Demo Recording Script for Assignment Submission.

Records a video demonstrating the squat analyzer functionality:
- Shows pose detection
- Shows angle calculations
- Shows corrective feedback
- Shows rep counting

Usage:
    python scripts/record_demo.py
    python scripts/record_demo.py --source test_videos/squat_proper_form.mp4
"""

import sys
import time
from pathlib import Path
import cv2
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from squat_analyzer.main import SquatAnalyzer
from squat_analyzer.config.settings import Settings


def main():
    parser = argparse.ArgumentParser(description="Record a demo video")
    parser.add_argument("--source", type=str, default=None, 
                        help="Video source (default: webcam)")
    parser.add_argument("--output", type=str, default="demo_output.mp4",
                        help="Output filename")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds (webcam only)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SQUAT ANALYZER - DEMO RECORDING")
    print("=" * 60)
    print(f"Source: {args.source or 'Webcam'}")
    print(f"Output: {args.output}")
    print("Press 'q' to stop recording")
    print("=" * 60)
    
    # Initialize
    settings = Settings()
    
    # Open video source
    if args.source:
        cap = cv2.VideoCapture(args.source)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open video source!")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Initialize analyzer components
    from squat_analyzer.core.pose_estimator import PoseEstimator
    from squat_analyzer.core.angles import AngleCalculator
    from squat_analyzer.analysis.biomechanics import BiomechanicsEngine
    from squat_analyzer.analysis.squat_detector import SquatDetector
    from squat_analyzer.visualization.renderer import OverlayRenderer
    
    pose_est = PoseEstimator(config=settings.pose)
    angle_calc = AngleCalculator()
    biomech = BiomechanicsEngine(config=settings.rules)
    squat_det = SquatDetector()
    renderer = OverlayRenderer(config=settings.visualization)
    
    frame_count = 0
    start_time = time.time()
    max_frames = int(args.duration * fps) if not args.source else float('inf')
    
    print("\nRecording started... Press 'q' to stop.\n")
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            keypoints = pose_est.estimate_single(frame)
            
            if keypoints is not None:
                # Get angles
                angles = angle_calc.compute_all_angles(keypoints)
                
                # Get biomechanical analysis
                results = biomech.analyze(keypoints)
                
                # Update squat detector
                phase = squat_det.update(keypoints)
                
                # Render visualization
                output_frame = renderer.render(
                    frame=frame,
                    keypoints=keypoints,
                    results=results,
                    phase=phase,
                    rep_count=squat_det.rep_count,
                    fps=fps,
                    angles=angles,
                )
            else:
                output_frame = renderer.render(
                    frame=frame,
                    fps=fps,
                    message="No person detected",
                )
            
            # Write frame
            out.write(output_frame)
            
            # Display
            cv2.imshow("Demo Recording (Press 'q' to stop)", output_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Recorded {frame_count} frames ({elapsed:.1f}s) - Reps: {squat_det.rep_count}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Recording complete!")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Reps detected: {squat_det.rep_count}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
