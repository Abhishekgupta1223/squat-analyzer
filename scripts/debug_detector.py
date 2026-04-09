"""Debug why squat detector stays in IDLE."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from squat_analyzer.core.pose_estimator import PoseEstimator
from squat_analyzer.core.angles import AngleCalculator
from squat_analyzer.analysis.squat_detector import SquatDetector, SquatPhase
from squat_analyzer.config.settings import Settings

def main():
    video_path = Path(__file__).parent.parent / "demo_videos" / "squat_proper_form.mp4"
    
    settings = Settings()
    pose_est = PoseEstimator(config=settings.pose)
    angle_calc = AngleCalculator()
    squat_det = SquatDetector()
    
    cap = cv2.VideoCapture(str(video_path))
    
    frame_num = 0
    max_frames = 600
    
    print("Frame | Knee° | Hip° | Torso° | Phase | Standing Valid?")
    print("-" * 60)
    
    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        keypoints = pose_est.estimate_single(frame)
        if keypoints is None:
            continue
        
        # Calculate angles manually to see what's happening
        knee = angle_calc.knee_flexion_angle(keypoints)
        hip = angle_calc.hip_flexion_angle(keypoints)
        torso = angle_calc.torso_inclination(keypoints)
        
        # Check standing conditions
        th = squat_det._thresholds
        knee_ok = knee >= th.standing_knee_angle  # >= 165
        hip_ok = hip >= th.standing_hip_angle      # >= 160
        torso_ok = torso <= th.standing_torso_max  # <= 30
        
        is_standing = knee_ok and hip_ok and torso_ok
        
        # Update detector
        phase = squat_det.update(keypoints)
        
        if frame_num <= 20 or frame_num % 20 == 0:
            print(f"{frame_num:5d} | {knee:5.1f} | {hip:5.1f} | {torso:6.1f} | {phase.name:10s} | "
                  f"knee={knee_ok}, hip={hip_ok}, torso={torso_ok} -> {is_standing}")
    
    cap.release()
    print(f"\nFinal reps: {squat_det.rep_count}")

if __name__ == "__main__":
    main()
