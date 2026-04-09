"""
Production-Grade Pose Estimation Module
=======================================

Provides a robust, high-performance pose estimation interface using
YOLOv8-pose models with automatic device selection, error handling,
and graceful degradation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from squat_analyzer.core.keypoints import Keypoints
from squat_analyzer.utils.logging import get_logger

if TYPE_CHECKING:
    from ultralytics import YOLO
    from squat_analyzer.config.settings import PoseEstimationConfig

logger = get_logger(__name__)


class PoseEstimationError(Exception):
    """Raised when pose estimation fails."""
    pass


class ModelLoadError(PoseEstimationError):
    """Raised when model loading fails."""
    pass


class PoseEstimator:
    """
    High-performance pose estimation using YOLOv8-pose.
    
    Features:
        - Automatic device selection (CUDA > MPS > CPU)
        - Configurable model size for speed/accuracy tradeoff
        - Robust error handling with graceful degradation
        - Support for batch processing
        - Half-precision inference for GPU acceleration
    
    Attributes:
        model_name: Name of the YOLOv8-pose model
        device: Inference device (cuda, mps, cpu)
        confidence_threshold: Minimum detection confidence
        
    Example:
        >>> from squat_analyzer.config import Settings
        >>> settings = Settings()
        >>> estimator = PoseEstimator(settings.pose)
        >>> keypoints = estimator.estimate(frame)
        >>> if keypoints:
        ...     print(f"Detected {len(keypoints)} people")
    """
    
    def __init__(
        self,
        config: Optional["PoseEstimationConfig"] = None,
        model_name: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        """
        Initialize the pose estimator.
        
        Args:
            config: PoseEstimationConfig instance (preferred)
            model_name: YOLOv8-pose model name (fallback if no config)
            confidence_threshold: Minimum confidence for detections
            device: Device for inference (auto, cpu, cuda, mps)
        """
        # Extract config values or use defaults
        if config is not None:
            self.model_name = f"{config.model.value}.pt"
            self.confidence_threshold = config.confidence_threshold
            self._device_preference = config.device
            self._half_precision = config.half_precision
        else:
            self.model_name = model_name
            self.confidence_threshold = confidence_threshold
            self._device_preference = device
            self._half_precision = False
        
        self._model: Optional["YOLO"] = None
        self._device: str = ""
        self._initialized: bool = False
        
        # Lazy initialization - model loaded on first use
        logger.info(
            "PoseEstimator initialized",
            model=self.model_name,
            device_preference=self._device_preference,
        )
    
    def _initialize(self) -> None:
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelLoadError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            ) from e
        
        # Resolve device
        self._device = self._resolve_device()
        
        # Load model
        logger.info("Loading pose model...", model=self.model_name, device=self._device)
        
        try:
            self._model = YOLO(self.model_name)
            
            # Apply half precision if GPU and enabled
            if self._half_precision and self._device != "cpu":
                self._model.half()
                logger.info("Half precision enabled")
            
            self._initialized = True
            logger.info("Model loaded successfully", device=self._device)
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}") from e
    
    def _resolve_device(self) -> str:
        """
        Resolve the inference device.
        
        Returns:
            Device string for PyTorch (cuda, mps, or cpu)
        """
        if self._device_preference != "auto":
            return self._device_preference
        
        # Auto-detect best available device
        try:
            import torch
            
            if torch.cuda.is_available():
                logger.info("CUDA available - using GPU acceleration")
                return "cuda"
            
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS available - using Apple Silicon acceleration")
                return "mps"
            
        except ImportError:
            pass
        
        logger.info("Using CPU for inference")
        return "cpu"
    
    def estimate(
        self,
        frame: NDArray[np.uint8],
        max_detections: int = 10,  # Support multiple people
    ) -> list[Keypoints]:
        """
        Estimate poses in a frame.
        
        Args:
            frame: Input image as BGR numpy array
            max_detections: Maximum number of people to detect (default 10 for multi-person)
            
        Returns:
            List of Keypoints objects, one per detected person
            
        Raises:
            PoseEstimationError: If estimation fails unrecoverably
        """
        # Ensure model is loaded
        if not self._initialized:
            self._initialize()
        
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []
        
        try:
            # Run inference
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                device=self._device,
                verbose=False,
                max_det=max_detections,
            )
            
            # Process results
            keypoints_list: list[Keypoints] = []
            
            for result in results:
                if result.keypoints is None:
                    continue
                
                # Get keypoint data
                kps_data = result.keypoints.data.cpu().numpy()
                
                for person_kps in kps_data:
                    # person_kps shape: (17, 3) with [x, y, confidence]
                    try:
                        keypoints = Keypoints(points=person_kps)
                        keypoints_list.append(keypoints)
                    except ValueError as e:
                        logger.debug(f"Invalid keypoints: {e}")
                        continue
            
            return keypoints_list
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return []
    
    def estimate_single(
        self,
        frame: NDArray[np.uint8],
    ) -> Optional[Keypoints]:
        """
        Estimate pose for a single person (most confident detection).
        
        This is optimized for squat analysis where we expect one person.
        
        Args:
            frame: Input image as BGR numpy array
            
        Returns:
            Keypoints for the most confident detection, or None
        """
        results = self.estimate(frame, max_detections=1)
        return results[0] if results else None
    
    def warmup(self, input_size: tuple[int, int] = (640, 480)) -> None:
        """
        Warm up the model with a dummy inference.
        
        This reduces latency on the first real inference.
        
        Args:
            input_size: Size of dummy input (width, height)
        """
        if not self._initialized:
            self._initialize()
        
        logger.info("Warming up model...")
        dummy = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        self.estimate(dummy)
        logger.info("Warmup complete")
    
    @property
    def device(self) -> str:
        """Current inference device."""
        if not self._initialized:
            self._initialize()
        return self._device
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is loaded."""
        return self._initialized
