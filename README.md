<p align="center">
  <img src="https://img.shields.io/badge/🏋️_Squat_Analyzer-Real--time_Form_Analysis-blue?style=for-the-badge" alt="Squat Analyzer"/>
</p>

<p align="center">
  <strong>AI-powered real-time squat form analysis using pose estimation and biomechanics</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-api">API</a> •
  <a href="#-benchmarks">Benchmarks</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/YOLOv8-pose-00FFFF?logo=yolo&logoColor=white" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
</p>

---

## Why Squat Analyzer?

Poor squat form leads to **injury** and **limited gains**. Personal trainers aren't always available. Existing apps give generic feedback.

**Squat Analyzer** uses computer vision to provide **real-time, biomechanics-based feedback** — like having a sports scientist watching every rep.

```
✅ "Excellent depth! Knee angle 95° is optimal"
❌ "Knees collapsing inward — push OUT over pinky toes"
```

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/Abhishekgupta1223/squat-analyzer.git
cd squat-analyzer
pip install -e .

# Run webapp
python run_webapp.py
```

Open **http://localhost:8000** → Upload a video or use webcam.

That's it. No configuration needed.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-person tracking** | Analyze multiple people simultaneously with independent rep counting |
| **6 biomechanical rules** | Research-backed evaluation (see [Technical Details](#-technical-details)) |
| **Real-time feedback** | Sub-100ms latency with adaptive frame throttling |
| **Signal processing** | One-Euro filter eliminates keypoint jitter |
| **Phase detection** | State machine tracks Standing → Descent → Bottom → Ascent |
| **Web interface** | No installation for end users — just open browser |

---

## 🔬 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Camera    │────▶│  YOLOv8-pose │────▶│  Biomechanics   │────▶│   Feedback   │
│   Input     │     │  17 keypoints│     │  Engine (6 rules)│     │   Generator  │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  One-Euro    │
                    │  Filter      │
                    └──────────────┘
```

### Pose Estimation
- **Model**: YOLOv8n-pose (2.4M parameters, ~25 FPS on CPU)
- **Output**: 17 COCO keypoints with confidence scores
- **Filtering**: Adaptive One-Euro filter (β=0.007, d_cutoff=1.0)

### Biomechanical Analysis

| Rule | Method | Threshold | Source |
|------|--------|-----------|--------|
| **Knee Flexion** | Hip-knee-ankle angle | 70°-135° optimal | Schoenfeld 2010 |
| **Knee Valgus** | Lateral deviation from hip-ankle line | <10° safe | Hewett 2005 |
| **Torso Angle** | Shoulder-hip vector vs vertical | 30°-75° | Escamilla 2001 |
| **Knee-Over-Toe** | Horizontal knee extension ratio | <15% forward | Fry 2003 |
| **Hip Hinge** | Torso-thigh angle | 45°-100° | Hartmann 2013 |
| **Depth Check** | Hip-to-ankle vertical ratio | >0.5 (parallel) | NSCA |

### State Machine

```python
STANDING ──(ratio < 0.85)──▶ DESCENDING
DESCENDING ──(ratio < 0.6)──▶ BOTTOM  
BOTTOM ──(velocity > 0)──▶ ASCENDING
ASCENDING ──(ratio > 0.9)──▶ STANDING  # Rep complete!
```

Hysteresis thresholds prevent false transitions from noise.

---

## 📊 Benchmarks

| Configuration | FPS | Latency | Memory |
|--------------|-----|---------|--------|
| CPU (i7-10750H) | 22 | 45ms | 480MB |
| GPU (RTX 3060) | 68 | 15ms | 1.2GB |
| WebSocket (localhost) | 18 | 68ms | 520MB |

*Measured with 640px frame resize on 1080p input.*

---

## 🏗️ Architecture

```
squat-analyzer/
├── src/squat_analyzer/
│   ├── core/                 # Pose estimation, keypoint handling, angle math
│   │   ├── pose_estimator.py # YOLOv8 wrapper with confidence filtering
│   │   ├── keypoints.py      # COCO keypoint mapping & extraction
│   │   └── angles.py         # Vector math for joint angles
│   │
│   ├── analysis/             # Biomechanical intelligence
│   │   ├── biomechanics.py   # 6 rule classes (Strategy pattern)
│   │   ├── squat_detector.py # Phase state machine
│   │   ├── feedback.py       # Priority-based message generation
│   │   └── multi_person_tracker.py  # Zone-locking tracker
│   │
│   ├── filtering/            # Signal processing
│   │   └── one_euro.py       # Adaptive noise filter
│   │
│   └── visualization/        # Rendering
│       └── renderer.py       # Skeleton overlay with feedback
│
├── webapp/                   # Web interface
│   ├── server.py             # FastAPI + WebSocket
│   └── static/index.html     # Single-page app
│
└── demo_videos/              # Test videos included
```

---

## 🔌 API

### Python

```python
from squat_analyzer.core import PoseEstimator
from squat_analyzer.analysis import BiomechanicsEngine, SquatDetector

# Initialize
pose = PoseEstimator()
engine = BiomechanicsEngine()
detector = SquatDetector()

# Process frame
keypoints = pose.estimate(frame)
angles = engine.compute_angles(keypoints)
results = engine.evaluate(keypoints, angles)
phase, reps = detector.update(angles)

# Get feedback
for r in results:
    if r.status == "FAIL":
        print(f"⚠️ {r.message}")
        print(f"   Fix: {r.correction}")
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analyze/{session_id}');

// Send frame as base64
ws.send(JSON.stringify({ frame: base64Frame }));

// Receive analysis
ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    console.log(data.feedback);  // Array of feedback messages
    console.log(data.reps);      // Current rep count
    console.log(data.phase);     // STANDING, DESCENDING, etc.
};
```

---

## ⚙️ Configuration

Edit `config/settings.yaml`:

```yaml
pose_estimation:
  model: yolov8n-pose.pt
  confidence_threshold: 0.5
  max_detections: 10

biomechanics:
  knee_flexion:
    optimal_range: [70, 135]
  knee_valgus:
    warning_threshold: 5
    fail_threshold: 10
  
filtering:
  one_euro:
    min_cutoff: 1.0
    beta: 0.007
```

---

## 📋 Assumptions & Limitations

<details>
<summary><strong>Click to expand</strong></summary>

### Assumptions
- **Camera**: Static, frontal or side view, 2-4m distance
- **Lighting**: Adequate ambient light (~100+ lux)
- **Subject**: Full body visible, fitted clothing
- **Movement**: Controlled tempo (>250ms per phase)

### Limitations
- **2D only**: Cannot detect rotation/twist
- **Single exercise**: Optimized for squats only
- **No load detection**: Barbell position not considered
- **Individual variation**: Thresholds based on average anthropometry

### Edge Cases
- Very tall/short individuals may need threshold calibration
- Wide-stance sumo squats reduce valgus detection accuracy
- Heeled weightlifting shoes may trigger false knee-over-toe warnings

</details>

---

## 📚 References

1. Schoenfeld, B.J. (2010). *Squatting Kinematics and Kinetics*. Journal of Strength and Conditioning Research.
2. Hewett, T.E. et al. (2005). *Biomechanical Measures of Neuromuscular Control*. American Journal of Sports Medicine.
3. Escamilla, R.F. (2001). *Knee biomechanics of the dynamic squat exercise*. Medicine & Science in Sports & Exercise.
4. Casiez, G. et al. (2012). *1€ Filter: A Simple Speed-based Low-pass Filter*. ACM CHI.

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -e ".[dev]"
pytest                    # Run tests
ruff check src/ tests/   # Lint
mypy src/                # Type check
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with ❤️ using YOLOv8, FastAPI, and sports science research</sub>
</p>
