# Squat Analyzer - Real-Time Posture Analysis System

A real-time computer vision system that analyzes squat form and provides corrective feedback using YOLOv8 pose estimation and biomechanics-based evaluation rules.

---

## 📋 Assignment Deliverables Checklist

| Requirement | Status | Details |
|-------------|--------|---------|
| **Identify & track human body** | ✅ | YOLOv8-pose with multi-person tracking |
| **Extract key body points** | ✅ | 17 COCO keypoints (shoulders, hips, knees, ankles) |
| **2-3 evaluation rules** | ✅ **Exceeded (6 rules)** | See [Computational Rules](#-computational-rules) |
| **Correct/Incorrect feedback** | ✅ | Visual overlays + text messages |
| **Demo interface** | ✅ | Web application at localhost:8000 |
| **README: Technical approach** | ✅ | This document |
| **README: Assumptions** | ✅ | See [Assumptions](#-assumptions) |
| **README: Limitations** | ✅ | See [Limitations](#-limitations) |
| **README: How to run** | ✅ | See [Quick Start](#-quick-start) |
| **Demo video** | 📹 | [Link to video] |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam (optional) or test videos (included)

### Installation

```bash
# Clone repository
git clone https://github.com/Abhishekgupta1223/squat-analyzer.git
cd squat-analyzer

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .
```

### Run Web Application

```bash
python run_webapp.py
```

Then open **http://localhost:8000** in your browser.

**Features:**
- Upload video files for analysis
- Use webcam for real-time feedback
- 9 demo videos included for testing

---

## 🔬 Technical Approach

### 1. Pose Estimation
- **Model:** YOLOv8-pose (nano variant) from Ultralytics
- **Output:** 17 COCO keypoints per detected person
- **Performance:** ~15-25 FPS on CPU, ~50-80 FPS on GPU

### 2. Keypoints Used
| Index | Keypoint | Purpose |
|-------|----------|---------|
| 5, 6 | Left/Right Shoulder | Torso angle calculation |
| 11, 12 | Left/Right Hip | Hip flexion, depth reference |
| 13, 14 | Left/Right Knee | Knee angle, valgus detection |
| 15, 16 | Left/Right Ankle | Depth ratio, knee-over-toe |

### 3. Signal Processing
- **One-Euro Filter:** Adaptive low-pass filter to reduce keypoint jitter
- **Hysteresis State Machine:** Prevents false phase transitions using dual thresholds

### 4. Squat Phase Detection
```
STANDING → DESCENDING → BOTTOM → ASCENDING → STANDING (rep complete)
```

---

## 📐 Computational Rules

### Rule 1: Knee Flexion (Squat Depth)
**Method:** Angular measurement between hip-knee-ankle vectors

```
angle = arccos(dot(hip→knee, ankle→knee) / (|hip→knee| × |ankle→knee|))
```

| Evaluation | Threshold |
|------------|-----------|
| ✅ Optimal depth | 70° - 135° |
| ⚠️ Too shallow | > 135° |
| ⚠️ Excessive depth | < 70° |

**Reference:** Schoenfeld (2010) - Squatting Kinematics and Kinetics

---

### Rule 2: Torso Inclination (Back Angle)
**Method:** Angle between torso vector and vertical axis

```
torso = shoulder_midpoint - hip_midpoint
angle = arccos(dot(torso, vertical) / |torso|)
```

| Evaluation | Threshold |
|------------|-----------|
| ✅ Good posture | 30° - 75° |
| ⚠️ Forward lean | > 75° |

**Reference:** Escamilla (2001) - Knee Biomechanics of Dynamic Squat

---

### Rule 3: Knee Valgus (Knee Collapse)
**Method:** Lateral deviation of knees relative to hip-ankle line

```
deviation = perpendicular_distance(knee, hip_ankle_line)
valgus_angle = arctan(deviation / hip_ankle_distance)
```

| Evaluation | Threshold |
|------------|-----------|
| ✅ Tracking well | < 5° |
| ⚠️ Slight valgus | 5° - 10° |
| ❌ Injury risk | > 10° |

**Reference:** Hewett et al. (2005) - Biomechanical Measures of Neuromuscular Control

---

### Rule 4: Knee-Over-Toe Position
**Method:** Horizontal distance ratio

```
forward_extension = (knee_x - ankle_x) / (hip_x - ankle_x)
```

| Evaluation | Threshold |
|------------|-----------|
| ✅ Safe position | < 15% extension |
| ⚠️ Moderate forward | 15% - 25% |

**Reference:** Fry et al. (2003) - Effect of Knee Position on Torques

---

### Rule 5: Hip Hinge Initiation
**Method:** Hip flexion angle measurement

```
angle = arccos(dot(torso, thigh) / (|torso| × |thigh|))
```

| Evaluation | Threshold |
|------------|-----------|
| ✅ Proper hinge | 45° - 100° |

**Reference:** Hartmann et al. (2013) - Load Analysis on Knee Joint

---

### Rule 6: Depth Verification (Hip-to-Knee Ratio)
**Method:** Vertical position ratio using shoulder-ankle reference

```
depth_ratio = (hip_y - shoulder_y) / (ankle_y - shoulder_y)
```

| Evaluation | Threshold |
|------------|-----------|
| ✅ Parallel or below | ratio > 0.5 |

**Reference:** NSCA Guidelines for Exercise Technique

---

## 💬 Feedback Examples

### Correct Posture
```
✅ "Excellent depth! Knee angle 95° is optimal"
✅ "Good torso position - spine neutral"  
✅ "Knees tracking well over toes"
```

### Incorrect Posture (with corrections)
```
❌ "Squat depth insufficient - hips too high"
   → Correction: "Sit back and down. Imagine sitting into a low chair"

❌ "⚠️ KNEE VALGUS - Injury risk! Knees collapsing inward"
   → Correction: "Push knees OUT over pinky toes"

❌ "Leaning forward - engage your core"
   → Correction: "Chest proud - lift sternum toward ceiling"
```

---

## ⚠️ Assumptions

### Camera & Environment
| Assumption | Rationale |
|------------|-----------|
| **Single-plane view** | Frontal or side camera view required; oblique angles reduce accuracy |
| **Adequate lighting** | Minimum ~100 lux for reliable keypoint detection |
| **Static camera** | Handheld footage causes tracking instability |
| **Full-body visibility** | Subject must be fully visible head to feet |
| **Distance 2-4 meters** | Too close clips body; too far reduces precision |

### Subject & Movement
| Assumption | Rationale |
|------------|-----------|
| **Upright starting position** | Calibration assumes ~180° knee angle when standing |
| **Controlled tempo** | Movements faster than 250ms per phase may be filtered as noise |
| **Standard proportions** | Thresholds based on average adult anthropometry |
| **Minimal occlusion** | Baggy clothing or crossed arms degrade detection |

---

## 🚧 Limitations

### Accuracy Limitations
| Limitation | Impact |
|------------|--------|
| **2D estimation only** | Cannot detect rotation or twist |
| **Keypoint jitter** | Filtered but not eliminated |
| **Occlusion sensitivity** | Equipment/hands can hide joints |
| **Lighting dependent** | Low light causes failures |

### Biomechanical Limitations
| Limitation | Impact |
|------------|--------|
| **Individual variation** | Optimal angles vary by anatomy |
| **Single movement type** | Tuned for squats only |
| **No load detection** | Barbell position not considered |
| **Fatigue not tracked** | No rep-over-rep trend analysis |

### System Limitations
| Limitation | Impact |
|------------|--------|
| **CPU-bound** | ~15-25 FPS without GPU |
| **Network latency** | WebSocket adds ~50-100ms |
| **Visual feedback only** | No audio output |

---

## 📁 Project Structure

```
squat-analyzer/
├── src/squat_analyzer/
│   ├── core/              # Pose estimation, keypoints, angles
│   ├── analysis/          # Biomechanics rules, squat detection
│   ├── filtering/         # One-Euro signal filter
│   └── visualization/     # Skeleton overlay rendering
├── webapp/                # FastAPI server + web UI
├── demo_videos/           # 9 test videos included
├── tests/                 # Unit tests
├── README.md              # This file
└── TECHNICAL_DOCUMENTATION.md  # Detailed technical deep-dive
```

---

## 📚 References

1. Schoenfeld, B.J. (2010). *Squatting Kinematics and Kinetics*. JSCR.
2. Hewett, T.E. et al. (2005). *Biomechanical Measures of Neuromuscular Control*. AJSM.
3. Escamilla, R.F. (2001). *Knee biomechanics of the dynamic squat*. MSSE.
4. Hartmann, H. et al. (2013). *Load Analysis on Knee Joint*. Sports Medicine.
5. Fry, A.C. et al. (2003). *Effect of Knee Position on Torques*. JSCR.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

*For detailed implementation specifics, see [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)*
