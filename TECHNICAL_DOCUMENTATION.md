# Squat Analyzer - Technical Documentation

## Overview

This document describes the computational methods and rules used in the Squat Analyzer system for evaluating squat posture.

---

## 1. Pose Estimation

**Model:** YOLOv8-pose (nano variant)  
**Framework:** Ultralytics  
**Keypoints:** 17 COCO body keypoints

### Extracted Keypoints for Squat Analysis:
| Index | Keypoint | Usage |
|-------|----------|-------|
| 5, 6 | Shoulders (L/R) | Torso angle |
| 11, 12 | Hips (L/R) | Hip flexion, stance width |
| 13, 14 | Knees (L/R) | Knee flexion, valgus |
| 15, 16 | Ankles (L/R) | Depth reference, knee-over-toe |

---

## 2. Computational Rules

### Rule 1: Torso Inclination (Angle Measurement)

**Purpose:** Ensure proper upright posture during squat

**Computation:**
```
torso_vector = hip_midpoint - shoulder_midpoint
vertical = [0, 1]  # Downward in image coordinates
angle = arccos(dot(torso_vector, vertical) / (|torso_vector| * |vertical|))
```

**Thresholds:**
- PASS: 0° - 30° (upright)
- WARNING: 30° - 45° (slight lean)
- FAIL: > 45° (excessive forward lean)

**Reference:** Escamilla (2001) - Biomechanics of the knee during closed kinetic chain exercises

---

### Rule 2: Knee Flexion Angle (Angle Measurement)

**Purpose:** Measure squat depth for proper muscle activation

**Computation:**
```
vec_hip_knee = knee - hip
vec_ankle_knee = knee - ankle
angle = arccos(dot(vec_hip_knee, vec_ankle_knee) / (|vec_hip_knee| * |vec_ankle_knee|))
```

**Thresholds:**
- Optimal range: 70° - 135°
- Below 70°: Excessive depth (mobility concern)
- Above 135°: Insufficient depth

**Reference:** Schoenfeld (2010) - Squatting kinematics and kinetics

---

### Rule 3: Knee Valgus Detection (Distance/Angle Measurement)

**Purpose:** Detect dangerous inward knee collapse

**Computation:**
```
# Project knee position relative to hip-ankle line
hip_ankle_line = ankle - hip
knee_offset = knee - ankle
projection = dot(knee_offset, normalize(hip_ankle_line))
perpendicular_distance = |knee_offset - projection * normalize(hip_ankle_line)|

# Positive = valgus (inward), Negative = varus (outward)
valgus_angle = arctan(perpendicular_distance / |hip_ankle_line|)
```

**Thresholds:**
- PASS: < 5° valgus
- WARNING: 5° - 10° valgus
- FAIL: > 10° valgus (injury risk)

**Reference:** Hewett et al. (2005) - Biomechanical measures of neuromuscular control

---

### Rule 4: Knee-Over-Toe Position (Distance Ratio)

**Purpose:** Ensure knees don't extend too far past toes

**Computation:**
```
knee_x = (left_knee_x + right_knee_x) / 2
toe_x = (left_ankle_x + right_ankle_x) / 2
hip_x = (left_hip_x + right_hip_x) / 2

# Ratio of knee extension relative to base
ratio = (knee_x - toe_x) / (hip_x - toe_x)
```

**Thresholds:**
- PASS: ratio < 1.0 (knee behind or at toes)
- WARNING: 1.0 - 1.15 (slight forward)
- FAIL: > 1.15 (excessive forward position)

**Reference:** Fry et al. (2003) - Effect of knee position on hip and knee torques

---

### Rule 5: Hip Shift (Distance Measurement)

**Purpose:** Detect lateral imbalance during squat

**Computation:**
```
left_hip_y = left_hip[1]
right_hip_y = right_hip[1]
shift_pixels = |left_hip_y - right_hip_y|
shift_ratio = shift_pixels / torso_length
```

**Thresholds:**
- PASS: shift_ratio < 5%
- WARNING: 5% - 10%
- FAIL: > 10% (significant asymmetry)

**Reference:** Movement assessment protocols

---

### Rule 6: Squat Depth Analysis (Combined Measurement)

**Purpose:** Verify proper squat depth is achieved

**Computation:**
```
# Method 1: Knee angle threshold
depth_by_angle = knee_angle <= 100°

# Method 2: Thigh parallel to ground
thigh_vector = knee - hip
horizontal = [1, 0]
thigh_angle = angle_between(thigh_vector, horizontal)
parallel = thigh_angle < 20°

# Combined assessment
valid_depth = depth_by_angle OR parallel
```

**Thresholds:**
- PASS: Thigh parallel or below
- WARNING: Above parallel (quarter squat)
- FAIL: Minimal depth (standing)

**Reference:** NSCA Strength & Conditioning Guidelines

---

## 3. Rep Counting Algorithm

### State Machine:
```
IDLE → STANDING → DESCENDING → BOTTOM → ASCENDING → STANDING
```

### Valid Rep Requirements:
1. **Start from standing:** knee > 160°, hip > 155°, torso < 30°
2. **Reach depth:** knee < 110° AND hip < 130°
3. **Return to standing:** knee > 160°
4. **Minimum duration:** > 500ms

### View-Adaptive Detection:
- **Side view:** Uses joint angles (traditional)
- **Front view:** Uses hip Y-position drop (position-based)

---

## 4. Signal Processing

### One-Euro Filter
Applied to keypoint positions to reduce jitter while maintaining responsiveness.

**Parameters:**
- min_cutoff: 1.0 Hz (smoothness)
- beta: 0.007 (speed coefficient)
- d_cutoff: 1.0 Hz (derivative cutoff)

**Reference:** Casiez et al. (2012) - 1€ Filter: A Simple Speed-based Low-pass Filter

---

## 5. References

1. Schoenfeld, B.J. (2010). Squatting kinematics and kinetics and their application to exercise performance.
2. Hewett, T.E. et al. (2005). Biomechanical measures of neuromuscular control and valgus loading of the knee.
3. Escamilla, R.F. (2001). Knee biomechanics of the dynamic squat exercise.
4. Hartmann, H. et al. (2013). Analysis of the load on the knee joint and vertebral column.
5. Fry, A.C. et al. (2003). Effect of knee position on hip and knee torques during the barbell squat.
6. Casiez, G. et al. (2012). 1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input.
