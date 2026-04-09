"""
FastAPI WebSocket Server for Real-time Squat Analysis
======================================================
Production-grade server with WebSocket streaming, session management,
and efficient frame processing pipeline.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

from squat_analyzer.config.settings import Settings
from squat_analyzer.core.pose_estimator import PoseEstimator
from squat_analyzer.core.keypoints import Keypoints
from squat_analyzer.analysis.biomechanics import BiomechanicsEngine, RuleStatus
from squat_analyzer.analysis.squat_detector_v2 import SquatDetectorV2, SquatPhase
from squat_analyzer.analysis.multi_person_tracker import MultiPersonTracker
from squat_analyzer.analysis.feedback import FeedbackGenerator
from squat_analyzer.visualization.renderer import OverlayRenderer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkoutSession:
    """Tracks a single workout session."""
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    rep_count: int = 0
    total_frames: int = 0
    form_scores: list = field(default_factory=list)
    feedback_history: list = field(default_factory=list)
    squat_phases: list = field(default_factory=list)
    
    def add_form_score(self, score: float):
        self.form_scores.append(score)
    
    def get_average_score(self) -> float:
        if not self.form_scores:
            return 0.0
        # Use recent scores (last 100)
        recent = self.form_scores[-100:]
        return sum(recent) / len(recent)
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "rep_count": self.rep_count,
            "total_frames": self.total_frames,
            "average_form_score": self.get_average_score(),
            "feedback_history": self.feedback_history[-10:],
        }


class AnalyzerPipeline:
    """Encapsulates the complete analysis pipeline with multi-person tracking."""
    
    # Target processing resolution (balance speed vs accuracy)
    TARGET_WIDTH = 640
    
    def __init__(self):
        logger.info("Initializing analysis pipeline...")
        
        self.settings = Settings()
        self.pose_estimator = PoseEstimator(config=self.settings.pose)
        self.biomechanics = BiomechanicsEngine(config=self.settings.rules)
        self.person_tracker = MultiPersonTracker()  # Multi-person tracking!
        self.feedback_generator = FeedbackGenerator()
        self.renderer = OverlayRenderer(config=self.settings.visualization)
        
        # Separate squat detector per tracked person
        self._detectors: dict[int, SquatDetectorV2] = {}
        
        # Latency tracking
        self._last_process_time: float = 0.0
        self._frame_times: list = []
        
        logger.info("Analysis pipeline initialized (multi-person mode)!")
    
    def reset(self):
        """Reset pipeline for new session."""
        self._detectors.clear()
        self.person_tracker.reset()
    
    def _get_detector(self, track_id: int) -> SquatDetectorV2:
        """Get or create detector for a tracked person."""
        if track_id not in self._detectors:
            self._detectors[track_id] = SquatDetectorV2()
            logger.info(f"Created detector for person #{track_id}")
        return self._detectors[track_id]
    
    def _get_total_reps(self) -> int:
        """Get total reps across all tracked people."""
        return sum(d.rep_count for d in self._detectors.values())
    
    def _get_per_person_stats(self) -> list[dict]:
        """Get stats per tracked person."""
        stats = []
        for track_id, detector in sorted(self._detectors.items()):
            stats.append({
                "person_id": track_id,
                "reps": detector.rep_count,
                "phase": detector.phase.name.lower() if hasattr(detector.phase, 'name') else str(detector.phase)
            })
        return stats
    
    def process_frame(self, frame: np.ndarray, session: WorkoutSession) -> tuple[np.ndarray, dict]:
        """Process a single frame and return annotated frame + analysis data."""
        start_time = time.time()
        
        try:
            # Resize frame for faster processing (maintain aspect ratio)
            h, w = frame.shape[:2]
            if w > self.TARGET_WIDTH:
                scale = self.TARGET_WIDTH / w
                new_w = self.TARGET_WIDTH
                new_h = int(h * scale)
                frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame
                scale = 1.0
                new_h, new_w = h, w
            
            # Pose estimation on smaller frame
            keypoints_list = self.pose_estimator.estimate(frame_small)
            
            # Check if any person was detected
            if not keypoints_list:
                self._track_latency(start_time)
                return frame, {"status": "no_person", "message": "No person detected", "latency_ms": self._last_process_time}
            
            # Multi-person tracking using SMALL frame dimensions
            tracked_persons = self.person_tracker.select_persons(keypoints_list, new_w, new_h)
            
            if not tracked_persons:
                self._track_latency(start_time)
                return frame, {"status": "no_person", "message": "No valid person detected", "latency_ms": self._last_process_time}
            
            # Scale keypoints back up to original resolution for rendering
            if scale != 1.0:
                for track_id, kp in tracked_persons.items():
                    if hasattr(kp, 'points') and kp.points is not None:
                        kp.points[:, :2] = kp.points[:, :2] / scale
            
            # Process EACH tracked person
            all_violations = []
            all_feedback = []
            annotated = frame.copy()
            
            for track_id, keypoints in tracked_persons.items():
                # Get detector for this person
                detector = self._get_detector(track_id)
                
                # Check leg visibility
                leg_indices = [11, 12, 13, 14, 15, 16]
                upper_indices = [0, 1, 2, 3, 4, 5, 6]
                
                if keypoints.confidence is not None:
                    leg_conf = [float(keypoints.confidence[i]) for i in leg_indices if i < len(keypoints.confidence)]
                    upper_conf = [float(keypoints.confidence[i]) for i in upper_indices if i < len(keypoints.confidence)]
                else:
                    leg_conf = [1.0] * len(leg_indices)
                    upper_conf = [1.0] * len(upper_indices)
                
                avg_leg = sum(leg_conf) / len(leg_conf) if leg_conf else 0
                avg_upper = sum(upper_conf) / len(upper_conf) if upper_conf else 0
                
                # Skip person if legs not visible
                if avg_upper > 0.5 and avg_leg < 0.05:
                    annotated = self.renderer.render(annotated, keypoints)
                    continue
                
                # Biomechanical analysis for this person
                rule_results = self.biomechanics.analyze(keypoints)
                
                # Squat detection for THIS person
                phase = detector.update(keypoints)
                
                # Collect violations
                violations = [r for r in rule_results if r.status in (RuleStatus.WARNING, RuleStatus.FAIL)]
                all_violations.extend(violations)
                
                # Generate feedback
                feedback_messages = self.feedback_generator.generate(rule_results, phase, detector.rep_count)
                all_feedback.extend(feedback_messages or [])
                
                # Render this person's pose
                annotated = self.renderer.render(
                    annotated, keypoints, rule_results, feedback_messages, phase, detector.rep_count
                )
            
            # Aggregate stats
            total_reps = self._get_total_reps()
            per_person_stats = self._get_per_person_stats()
            person_count = self.person_tracker.get_track_count()
            
            # Update session with TOTAL reps
            if total_reps > session.rep_count:
                session.rep_count = total_reps
            
            # Calculate form score
            if all_violations:
                severity_map = {RuleStatus.WARNING: 0.8, RuleStatus.FAIL: 0.5}
                min_score = min(severity_map.get(v.status, 0.5) for v in all_violations)
                session.add_form_score(min_score)
            else:
                session.add_form_score(1.0)
            
            # Store feedback
            for msg in all_feedback:
                text = msg.text if hasattr(msg, 'text') else str(msg)
                if text not in session.feedback_history:
                    session.feedback_history.append(text)
            
            session.total_frames += 1
            avg_score = session.get_average_score()
            
            # Extract feedback messages
            feedback_texts = []
            if all_feedback:
                feedback_texts = [
                    msg.text if hasattr(msg, 'text') else str(msg) 
                    for msg in all_feedback[:3]
                ]
            if not feedback_texts:
                feedback_texts = ["Form looks good!"]
            
            # Build violations list
            violation_list = []
            for r in all_violations[:3]:
                violation_list.append({
                    "rule": r.name,
                    "severity": "high" if r.status == RuleStatus.FAIL else "medium",
                    "message": r.message
                })
            
            self._track_latency(start_time)
            
            return annotated, {
                "status": "analyzing",
                "rep_count": total_reps,
                "person_count": person_count,
                "per_person": per_person_stats,
                "phase": per_person_stats[0]["phase"] if per_person_stats else "idle",
                "form_score": int(avg_score * 100),
                "current_feedback": feedback_texts,
                "violations": violation_list,
                "latency_ms": self._last_process_time
            }
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            return frame, {"status": "error", "message": str(e)}
    
    def _track_latency(self, start_time: float):
        """Track processing latency for monitoring."""
        elapsed = (time.time() - start_time) * 1000  # ms
        self._last_process_time = round(elapsed, 1)
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 100:
            self._frame_times = self._frame_times[-100:]
    
    def get_avg_latency(self) -> float:
        """Get average processing latency in ms."""
        if not self._frame_times:
            return 0.0
        return sum(self._frame_times) / len(self._frame_times)


# Global pipeline instance (initialized on startup)
pipeline: Optional[AnalyzerPipeline] = None
sessions: dict[str, WorkoutSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global pipeline
    logger.info("Starting Squat Analyzer Web Server...")
    pipeline = AnalyzerPipeline()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Squat Analyzer Pro",
    description="Real-time AI-powered squat form analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Demo videos for testing
demo_videos_path = Path(__file__).parent.parent / "demo_videos"
if demo_videos_path.exists():
    app.mount("/demo-videos", StaticFiles(directory=str(demo_videos_path)), name="demo-videos")


@app.get("/api/demo-videos")
async def list_demo_videos():
    """List available demo videos."""
    videos = []
    if demo_videos_path.exists():
        for f in demo_videos_path.glob("*.mp4"):
            videos.append({
                "name": f.stem.replace("_", " ").title(),
                "filename": f.name,
                "url": f"/demo-videos/{f.name}",
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
            })
    return {"videos": videos}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Squat Analyzer Pro</h1><p>Static files not found.</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "pipeline_ready": pipeline is not None}


@app.post("/api/session/start")
async def start_session():
    """Start a new workout session."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = WorkoutSession(session_id=session_id)
    
    # Reset pipeline for new session
    if pipeline:
        pipeline.reset()  # Reset detector and person tracker
    
    logger.info(f"Started session: {session_id}")
    return {"session_id": session_id}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session statistics."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id].to_dict()


@app.post("/api/session/{session_id}/end")
async def end_session(session_id: str):
    """End a workout session and get final stats."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    stats = session.to_dict()
    
    logger.info(f"Ended session: {session_id}, reps: {session.rep_count}")
    
    return stats


@app.websocket("/ws/analyze/{session_id}")
async def websocket_analyze(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time video analysis."""
    await websocket.accept()
    logger.info(f"WebSocket connected for session: {session_id}")
    
    # Create session if doesn't exist
    if session_id not in sessions:
        sessions[session_id] = WorkoutSession(session_id=session_id)
    
    session = sessions[session_id]
    
    try:
        while True:
            # Receive frame data (base64 encoded JPEG)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                # Decode base64 image
                img_data = base64.b64decode(message["data"].split(",")[1] if "," in message["data"] else message["data"])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"type": "error", "message": "Invalid frame"})
                    continue
                
                # Process frame
                annotated, analysis = pipeline.process_frame(frame, session)
                
                # Encode result frame
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                result_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send response
                await websocket.send_json({
                    "type": "analysis",
                    "frame": f"data:image/jpeg;base64,{result_b64}",
                    "analysis": analysis,
                    "session": {
                        "rep_count": session.rep_count,
                        "form_score": int(session.get_average_score() * 100),
                        "duration": (datetime.now() - session.start_time).total_seconds()
                    }
                })
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze an uploaded video file."""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # Create temporary session
    session_id = str(uuid.uuid4())
    session = WorkoutSession(session_id=session_id)
    
    # Save uploaded file
    temp_path = Path(__file__).parent / "temp" / f"{session_id}_{file.filename}"
    temp_path.parent.mkdir(exist_ok=True)
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process video
        cap = cv2.VideoCapture(str(temp_path))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame for efficiency
            if frame_count % 3 == 0:
                _, _ = pipeline.process_frame(frame, session)
            
            frame_count += 1
        
        cap.release()
        
        return {
            "session_id": session_id,
            "frames_processed": frame_count,
            "stats": session.to_dict()
        }
        
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
