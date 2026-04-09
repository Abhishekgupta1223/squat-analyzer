# Squat Analyzer - Production-Grade Pose Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/license/MIT-blue.svg)](LICENSE)

A **production-grade, research-backed** real-time squat posture analysis system using state-of-the-art computer vision and biomechanical analysis.

## 🎯 Features

- **Real-time pose estimation** using YOLOv8-pose (17 keypoints at 30+ FPS)
- **6 research-validated biomechanical rules** based on sports science literature
- **Adaptive signal filtering** using One-Euro Filter for smooth tracking
- **State machine** for precise squat phase detection (Standing → Descending → Bottom → Ascending)
- **Priority-based feedback** with visual and text overlays
- **Production-ready** with Docker, CI/CD, comprehensive testing

## 📊 Biomechanical Analysis

| Rule | Description | Threshold | Reference |
|------|-------------|-----------|-----------|
| Knee Flexion | Optimal squat depth | 70° - 135° | Schoenfeld et al., 2010 |
| Knee Valgus | Prevent inward collapse | < 10° deviation | Hewett et al., 2005 |
| Torso Inclination | Maintain upright posture | 30° - 75° | Escamilla, 2001 |
| Hip Hinge | Proper hip mechanics | 45° - 100° | Hartmann et al., 2013 |
| Knee-Over-Toe | Safe knee positioning | < 15% extension | Fry et al., 2003 |
| Depth Analysis | Full ROM verification | Thigh parallel check | NSCA Guidelines |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/example/squat-analyzer.git
cd squat-analyzer

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"
```

### Run Analysis

```bash
# Webcam (default)
squat-analyzer

# Video file
squat-analyzer --source path/to/video.mp4

# Image
squat-analyzer --source path/to/image.jpg --mode image

# RTSP stream
squat-analyzer --source rtsp://camera/stream
```

### Docker

```bash
# Build image
docker build -t squat-analyzer .

# Run with webcam (Linux with X11)
docker run -it --rm \
    --device=/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    squat-analyzer
```

## 🏗️ Architecture

```
squat_analyzer/
├── src/
│   └── squat_analyzer/
│       ├── core/           # Pose estimation & angle computation
│       ├── analysis/       # Biomechanical rules & squat detection
│       ├── filtering/      # Signal processing (One-Euro Filter)
│       ├── visualization/  # Real-time overlays
│       ├── config/         # Pydantic settings management
│       └── utils/          # Logging, metrics, helpers
├── tests/                  # Comprehensive pytest suite
├── config/                 # YAML configuration files
├── docker/                 # Docker & compose files
└── docs/                   # Documentation
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=squat_analyzer --cov-report=html

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| FPS (YOLOv8n-pose, CPU) | ~15-25 |
| FPS (YOLOv8n-pose, GPU) | ~50-80 |
| Latency (detection) | ~20-40ms |
| Memory Usage | ~500MB |

---

## ⚠️ Assumptions & Limitations

### Assumptions

#### Camera & Environment
| Assumption | Rationale |
|------------|-----------|
| **Single-plane view** | System assumes frontal or sagittal (side) camera view. Oblique angles reduce accuracy. |
| **Adequate lighting** | Minimum ~100 lux ambient light required for reliable keypoint detection. |
| **Static camera** | Camera should be stationary; handheld footage causes tracking instability. |
| **Full-body visibility** | Subject must be fully visible from head to feet for accurate depth assessment. |
| **Distance 2-4 meters** | Optimal detection occurs at medium distance; too close clips body, too far reduces keypoint precision. |

#### Subject & Movement
| Assumption | Rationale |
|------------|-----------|
| **Upright starting position** | Calibration assumes ~180° knee angle at standing. Non-standard start positions may miscalibrate depth ratios. |
| **Controlled tempo** | Rapid/ballistic movements (< 250ms per phase) may be rejected as noise. |
| **Standard anthropometry** | Thresholds derived from average adult proportions; extreme body types may trigger false warnings. |
| **Minimal occlusion** | Baggy clothing, objects, or self-occlusion (crossed arms) degrades pose estimation. |

#### Technical Dependencies
| Assumption | Rationale |
|------------|-----------|
| **YOLOv8 keypoint accuracy** | Rules depend on COCO-format 17-keypoint detection; different models may require threshold recalibration. |
| **2D projection suffices** | 3D joint angles approximated from 2D projections; true 3D would require depth cameras or multi-view. |
| **Stable frame rate** | Filtering algorithms assume ~15-30 FPS input; severe frame drops cause detection discontinuities. |

---

### Limitations

#### Accuracy Constraints

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **2D pose estimation** | Cannot detect rotation/twist (e.g., asymmetric torso rotation) | Future: stereo cameras or IMU fusion |
| **Keypoint jitter** | Small oscillations in detection cause score fluctuation | One-Euro adaptive filtering applied |
| **Occlusion sensitivity** | Hands on hips, crossed arms, or equipment can hide keypoints | Confidence-based rejection implemented |
| **Lighting sensitivity** | Low light or backlighting causes detection failures | Recommend front-lighting setup |
| **Multi-person interference** | Overlapping subjects may cause track switching | Zone-locking tracker mitigates this |

#### Biomechanical Constraints

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Individual variation** | Optimal angles vary by anatomy (femur length, hip structure) | Configurable thresholds in `settings.yaml` |
| **Movement style agnostic** | High-bar vs low-bar squat have different optimal torso angles | Single rule set; advanced users can customize |
| **No load consideration** | Barbell position affects mechanics but isn't detected | Designed for bodyweight/light load assessment |
| **Fatigue not modeled** | Form degrades with fatigue; system doesn't track rep-over-rep changes | Future: longitudinal trend analysis |

#### System Constraints

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **CPU-bound inference** | ~15-25 FPS on CPU limits real-time feedback speed | GPU acceleration available; frame resizing to 640px |
| **Network latency (webapp)** | WebSocket round-trip adds ~50-100ms to feedback loop | Adaptive frame throttling implemented |
| **No audio feedback** | Visual-only output; users must watch screen | Future: text-to-speech integration |
| **Single exercise** | System tuned for squats only | Architecture extensible to other movements |

---

### Known Edge Cases

```
1. Very tall/short individuals → Depth ratios may need manual calibration
2. Wide-stance sumo squats → Valgus detection less reliable
3. Heeled shoes (weightlifting) → Knee-over-toe thresholds may be too strict
4. Pregnancy/injury accommodation → Thresholds not medically validated
5. Children → Model trained on adult proportions; accuracy unverified
```

---

### Recommendations for Best Results

1. **Camera placement**: Position camera at hip height, 2-3 meters away, perpendicular to movement plane
2. **Lighting**: Ensure even, front-facing light; avoid backlighting from windows
3. **Clothing**: Wear fitted clothing; avoid loose/baggy items that obscure joint positions
4. **Background**: Plain, uncluttered background improves detection confidence
5. **Warm-up**: Perform 2-3 calibration squats before formal assessment

---

## 📚 Research References

1. Schoenfeld, B.J. (2010). *Squatting Kinematics and Kinetics and Their Application to Exercise Performance*. JSCR.
2. Hewett, T.E. et al. (2005). *Biomechanical Measures of Neuromuscular Control*. AJSM.
3. Escamilla, R.F. (2001). *Knee biomechanics of the dynamic squat exercise*. MSSE.
4. Hartmann, H. et al. (2013). *Analysis of the Load on the Knee Joint and Vertebral Column*. Sports Medicine.
5. Fry, A.C. et al. (2003). *Effect of Knee Position on Hip and Knee Torques*. JSCR.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
