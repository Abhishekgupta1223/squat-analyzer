"""Quick test on single video to verify fixes."""
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

def main():
    video_path = Path(__file__).parent.parent / "test_videos" / "squat_proper_form.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
    
    print(f"Testing: {video_path.name}")
    
    settings = Settings()
    pose_est = PoseEstimator(config=settings.pose)
    angle_calc = AngleCalculator()
    biomech = BiomechanicsEngine(config=settings.rules)
    squat_det = SquatDetector()
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    # Stats
    person_detected = 0
    angles_computed = 0
    rules_triggered = 0
    phases = {"IDLE": 0, "STANDING": 0, "DESCENDING": 0, "BOTTOM": 0, "ASCENDING": 0}
    
    frame_num = 0
    max_frames = min(500, total_frames)  # Test first 500 frames only
    
    knee_angles = []
    
    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        keypoints = pose_est.estimate_single(frame)
        if keypoints is None:
            continue
        
        person_detected += 1
        
        # Compute angles
        angles = angle_calc.compute_all_angles(keypoints)
        knee = angles.get("knee_flexion", np.nan)
        
        if not np.isnan(knee):
            angles_computed += 1
            knee_angles.append(knee)
        
        # Biomechanics
        results = biomech.analyze(keypoints)
        if results:
            rules_triggered += 1
        
        # Phase
        phase = squat_det.update(keypoints)
        phases[phase.name] += 1
        
        if frame_num % 100 == 0:
            print(f"  Frame {frame_num}: knee={knee:.1f}°, phase={phase.name}, reps={squat_det.rep_count}")
    
    cap.release()
    
    print(f"\n=== RESULTS ===")
    print(f"Frames processed: {frame_num}")
    print(f"Person detected: {person_detected} ({100*person_detected/frame_num:.1f}%)")
    print(f"Angles computed: {angles_computed} ({100*angles_computed/max(1,person_detected):.1f}%)")
    print(f"Rules triggered: {rules_triggered} ({100*rules_triggered/max(1,person_detected):.1f}%)")
    print(f"Reps detected: {squat_det.rep_count}")
    print(f"\nPhase distribution:")
    for phase, count in phases.items():
        print(f"  {phase}: {count}")
    
    if knee_angles:
        print(f"\nKnee angle stats:")
        print(f"  Min: {min(knee_angles):.1f}°")
        print(f"  Max: {max(knee_angles):.1f}°")
        print(f"  Mean: {np.mean(knee_angles):.1f}°")

if __name__ == "__main__":
    main()
