"""
Test view-adaptive squat detection on webcam-style videos.

This tests both front-facing and side-view detection.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from squat_analyzer.core.pose_estimator import PoseEstimator
from squat_analyzer.core.angles import AngleCalculator
from squat_analyzer.analysis.biomechanics import BiomechanicsEngine
from squat_analyzer.analysis.squat_detector import SquatDetector
from squat_analyzer.config.settings import Settings


def test_video(video_path: Path, max_frames: int = 300):
    """Test a single video with view-adaptive detection."""
    print(f"\n{'='*60}")
    print(f"Testing: {video_path.name}")
    print(f"{'='*60}")
    
    settings = Settings()
    pose_est = PoseEstimator(config=settings.pose)
    squat_det = SquatDetector()
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}")
    
    frame_num = 0
    person_detected = 0
    phases = {"IDLE": 0, "STANDING": 0, "DESCENDING": 0, "BOTTOM": 0, "ASCENDING": 0}
    
    while frame_num < min(max_frames, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        keypoints = pose_est.estimate_single(frame)
        if keypoints is None:
            continue
        
        person_detected += 1
        phase = squat_det.update(keypoints)
        phases[phase.name] += 1
        
        # Print progress every 50 frames
        if frame_num % 50 == 0:
            print(f"  Frame {frame_num}: phase={phase.name}, reps={squat_det.rep_count}")
    
    cap.release()
    
    # Results
    print(f"\n📊 Results for {video_path.name}:")
    print(f"  Detected view: {squat_det._detected_view.upper()}")
    print(f"  Frames processed: {frame_num}")
    print(f"  Person detected: {person_detected} ({100*person_detected/max(1,frame_num):.1f}%)")
    print(f"  Reps detected: {squat_det.rep_count}")
    print(f"\n  Phase distribution:")
    for phase, count in phases.items():
        pct = 100 * count / max(1, person_detected)
        print(f"    {phase}: {count} ({pct:.1f}%)")
    
    return {
        "video": video_path.name,
        "view": squat_det._detected_view,
        "reps": squat_det.rep_count,
        "detection_rate": person_detected / max(1, frame_num),
    }


def main():
    test_dir = Path(__file__).parent.parent / "test_videos"
    
    # Test webcam-style videos first
    webcam_videos = [
        "webcam_squat_front.mp4",
        "home_squat_realtime.mp4",
        "laptop_squat_facing.mp4",
    ]
    
    # Then professional videos
    pro_videos = [
        "squat_proper_form.mp4",
        "bodyweight_squat.mp4",
    ]
    
    print("\n" + "#"*60)
    print("# VIEW-ADAPTIVE SQUAT DETECTION TEST")
    print("# Testing both front-facing and side-view videos")
    print("#"*60)
    
    results = []
    
    print("\n--- WEBCAM-STYLE VIDEOS (Front-facing) ---")
    for video_name in webcam_videos:
        video_path = test_dir / video_name
        if video_path.exists():
            result = test_video(video_path, max_frames=400)
            results.append(result)
    
    print("\n--- PROFESSIONAL VIDEOS (Side-view) ---")
    for video_name in pro_videos:
        video_path = test_dir / video_name
        if video_path.exists():
            result = test_video(video_path, max_frames=400)
            results.append(result)
    
    # Summary
    print("\n" + "#"*60)
    print("# SUMMARY")
    print("#"*60)
    
    print(f"\n{'Video':<30} {'View':<10} {'Reps':<8} {'Detection':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['video']:<30} {r['view']:<10} {r['reps']:<8} {r['detection_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
