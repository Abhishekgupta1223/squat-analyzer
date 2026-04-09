"""
Automated Video Testing Script for Squat Analyzer.

Tests the analyzer on real-world videos and generates comprehensive metrics.
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import traceback

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from squat_analyzer.core.pose_estimator import PoseEstimator
from squat_analyzer.core.angles import AngleCalculator
from squat_analyzer.analysis.biomechanics import BiomechanicsEngine
from squat_analyzer.analysis.squat_detector import SquatDetector
from squat_analyzer.config.settings import Settings


@dataclass
class VideoTestResult:
    """Results from testing a video."""
    video_name: str
    total_frames: int = 0
    processed_frames: int = 0
    person_detected_frames: int = 0
    legs_visible_frames: int = 0
    squat_reps_detected: int = 0
    avg_fps: float = 0.0
    errors: list = field(default_factory=list)
    rule_triggers: dict = field(default_factory=dict)
    angle_stats: dict = field(default_factory=dict)
    phases_detected: dict = field(default_factory=dict)
    

def analyze_video(video_path: Path, settings: Settings) -> VideoTestResult:
    """Analyze a video and return comprehensive metrics."""
    result = VideoTestResult(video_name=video_path.name)
    
    # Initialize components (matching main.py exactly)
    pose_estimator = PoseEstimator(config=settings.pose)
    angle_calculator = AngleCalculator()
    biomechanics = BiomechanicsEngine(config=settings.rules)
    squat_detector = SquatDetector()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.errors.append(f"Failed to open video: {video_path}")
        return result
    
    result.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tracking
    knee_angles = []
    hip_angles = []
    torso_angles = []
    frame_times = []
    
    # Initialize rule tracking
    rule_names = ["torso_upright", "knee_over_toe", "depth", "knee_valgus", "hip_shift", "stance_width"]
    for name in rule_names:
        result.rule_triggers[name] = {"pass": 0, "warning": 0, "fail": 0, "skip": 0}
    
    result.phases_detected = {"IDLE": 0, "STANDING": 0, "DESCENDING": 0, "BOTTOM": 0, "ASCENDING": 0}
    
    print(f"\n{'='*60}")
    print(f"Testing: {video_path.name}")
    print(f"Total frames: {result.total_frames}")
    print(f"{'='*60}")
    
    frame_num = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            frame_start = time.time()
            
            try:
                # Pose estimation
                keypoints = pose_estimator.estimate_single(frame)
                
                if keypoints is None:
                    continue
                
                result.person_detected_frames += 1
                
                # Check leg visibility
                upper_keypoints = [5, 6, 7, 8, 9, 10]
                leg_keypoints = [11, 12, 13, 14, 15, 16]
                
                upper_conf = np.mean([keypoints.get_confidence(i) for i in upper_keypoints])
                leg_conf = np.mean([keypoints.get_confidence(i) for i in leg_keypoints])
                
                # Lenient visibility check - only skip if clearly no legs
                legs_visible = leg_conf >= 0.1  # Very lenient
                
                if not legs_visible:
                    continue
                
                result.legs_visible_frames += 1
                
                # Calculate angles
                angles = angle_calculator.compute_all_angles(keypoints)
                
                # FIX: Use correct angle key names
                if "knee_flexion_left" in angles and not np.isnan(angles["knee_flexion_left"]):
                    knee_angles.append(angles["knee_flexion_left"])
                if "hip_flexion_left" in angles and not np.isnan(angles["hip_flexion_left"]):
                    hip_angles.append(angles["hip_flexion_left"])
                if "torso_inclination" in angles and not np.isnan(angles["torso_inclination"]):
                    torso_angles.append(angles["torso_inclination"])
                
                # Biomechanical analysis
                rules = biomechanics.analyze(keypoints)
                for rule in rules:
                    if rule.name in result.rule_triggers:
                        result.rule_triggers[rule.name][rule.status.name.lower()] += 1
                
                # Phase detection
                phase = squat_detector.update(keypoints)
                result.phases_detected[phase.name] += 1
                
                result.processed_frames += 1
                
            except Exception as e:
                result.errors.append(f"Frame {frame_num}: {str(e)}")
            
            frame_times.append(time.time() - frame_start)
            
            # Progress every 100 frames
            if frame_num % 100 == 0:
                print(f"  Processed {frame_num}/{result.total_frames} frames...")
    
    except Exception as e:
        result.errors.append(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
    
    finally:
        cap.release()
    
    # Calculate final stats
    total_time = time.time() - start_time
    result.squat_reps_detected = squat_detector.rep_count
    result.avg_fps = result.total_frames / total_time if total_time > 0 else 0
    
    # Angle statistics
    if knee_angles:
        result.angle_stats["knee"] = {
            "min": float(np.min(knee_angles)),
            "max": float(np.max(knee_angles)),
            "mean": float(np.mean(knee_angles)),
            "std": float(np.std(knee_angles)),
        }
    if hip_angles:
        result.angle_stats["hip"] = {
            "min": float(np.min(hip_angles)),
            "max": float(np.max(hip_angles)),
            "mean": float(np.mean(hip_angles)),
            "std": float(np.std(hip_angles)),
        }
    if torso_angles:
        result.angle_stats["torso"] = {
            "min": float(np.min(torso_angles)),
            "max": float(np.max(torso_angles)),
            "mean": float(np.mean(torso_angles)),
            "std": float(np.std(torso_angles)),
        }
    
    return result


def print_result(result: VideoTestResult) -> None:
    """Print formatted test results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {result.video_name}")
    print(f"{'='*60}")
    
    print(f"\n📊 Frame Statistics:")
    print(f"  Total frames:          {result.total_frames}")
    print(f"  Person detected:       {result.person_detected_frames} ({100*result.person_detected_frames/max(1,result.total_frames):.1f}%)")
    print(f"  Legs visible:          {result.legs_visible_frames} ({100*result.legs_visible_frames/max(1,result.total_frames):.1f}%)")
    print(f"  Successfully analyzed: {result.processed_frames} ({100*result.processed_frames/max(1,result.total_frames):.1f}%)")
    
    print(f"\n🏋️ Squat Detection:")
    print(f"  Reps detected: {result.squat_reps_detected}")
    
    print(f"\n📐 Phase Distribution:")
    for phase, count in result.phases_detected.items():
        pct = 100 * count / max(1, result.processed_frames)
        bar = "█" * int(pct / 5)
        print(f"  {phase:12s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    print(f"\n📏 Angle Statistics:")
    for name, stats in result.angle_stats.items():
        print(f"  {name.capitalize():6s}: min={stats['min']:.1f}°, max={stats['max']:.1f}°, mean={stats['mean']:.1f}°±{stats['std']:.1f}°")
    
    print(f"\n🔍 Rule Triggers (per analyzed frame):")
    for rule, counts in result.rule_triggers.items():
        total = sum(counts.values())
        if total > 0:
            pass_pct = 100 * counts['pass'] / total
            warn_pct = 100 * counts['warning'] / total
            fail_pct = 100 * counts['fail'] / total
            print(f"  {rule:15s}: ✅{pass_pct:5.1f}% | ⚠️{warn_pct:5.1f}% | ❌{fail_pct:5.1f}%")
    
    print(f"\n⚡ Performance:")
    print(f"  Average FPS: {result.avg_fps:.1f}")
    
    if result.errors:
        print(f"\n❗ Errors ({len(result.errors)}):")
        for err in result.errors[:5]:
            print(f"  - {err[:100]}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more")
    
    print()


def main():
    """Run tests on all videos in test_videos directory."""
    test_dir = Path(__file__).parent.parent / "test_videos"
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
    
    videos = list(test_dir.glob("*.mp4"))
    if not videos:
        print(f"No videos found in {test_dir}")
        return
    
    print(f"\n{'#'*60}")
    print(f"# SQUAT ANALYZER - AUTOMATED VIDEO TESTING")
    print(f"# Found {len(videos)} videos to test")
    print(f"{'#'*60}")
    
    settings = Settings()
    results = []
    
    for video_path in videos:
        result = analyze_video(video_path, settings)
        results.append(result)
        print_result(result)
    
    # Summary
    print(f"\n{'#'*60}")
    print(f"# SUMMARY")
    print(f"{'#'*60}")
    
    total_frames = sum(r.total_frames for r in results)
    total_analyzed = sum(r.processed_frames for r in results)
    total_reps = sum(r.squat_reps_detected for r in results)
    total_errors = sum(len(r.errors) for r in results)
    avg_fps = np.mean([r.avg_fps for r in results])
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Videos tested:     {len(results)}")
    print(f"  Total frames:      {total_frames}")
    print(f"  Frames analyzed:   {total_analyzed} ({100*total_analyzed/max(1,total_frames):.1f}%)")
    print(f"  Total reps found:  {total_reps}")
    print(f"  Total errors:      {total_errors}")
    print(f"  Average FPS:       {avg_fps:.1f}")
    
    # Save results to JSON
    output_path = test_dir / "test_results.json"
    with open(output_path, "w") as f:
        json.dump([{
            "video": r.video_name,
            "total_frames": r.total_frames,
            "processed_frames": r.processed_frames,
            "person_detected": r.person_detected_frames,
            "legs_visible": r.legs_visible_frames,
            "reps": r.squat_reps_detected,
            "fps": r.avg_fps,
            "errors": len(r.errors),
            "angle_stats": r.angle_stats,
            "phases": r.phases_detected,
            "rules": r.rule_triggers,
        } for r in results], f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
