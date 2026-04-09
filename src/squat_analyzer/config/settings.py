"""
Production-Grade Configuration System Using Pydantic v2
=======================================================

This module provides type-safe, validated configuration management
with environment variable support, YAML loading, and sensible defaults
based on biomechanical research literature.

Research References:
    - Schoenfeld (2010): Knee flexion angles for optimal muscle activation
    - Hewett et al. (2005): Knee valgus thresholds for injury prevention
    - Escamilla (2001): Torso inclination biomechanics
    - Hartmann et al. (2013): Hip hinge mechanics in loaded squats
    - Fry et al. (2003): Knee-over-toe positioning guidelines
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Logging level enumeration."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AnalysisMode(str, Enum):
    """Analysis mode enumeration."""
    
    REALTIME = "realtime"
    VIDEO = "video"
    IMAGE = "image"


class PoseModelType(str, Enum):
    """Supported pose estimation models."""
    
    YOLOV8N = "yolov8n-pose"
    YOLOV8S = "yolov8s-pose"
    YOLOV8M = "yolov8m-pose"
    YOLOV8L = "yolov8l-pose"
    YOLOV8X = "yolov8x-pose"


class RuleConfig(BaseModel):
    """Configuration for a single biomechanical rule."""
    
    enabled: bool = True
    severity: int = Field(default=5, ge=1, le=10)
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    description: str = ""

    @model_validator(mode="after")
    def validate_thresholds(self) -> "RuleConfig":
        """Ensure min < max when both are set."""
        if self.min_threshold is not None and self.max_threshold is not None:
            if self.min_threshold >= self.max_threshold:
                raise ValueError("min_threshold must be less than max_threshold")
        return self


class KneeFlexionConfig(RuleConfig):
    """Knee flexion angle configuration.
    
    Research: Schoenfeld (2010) - optimal muscle activation occurs
    at knee angles between 70° and 135°.
    """
    
    min_threshold: float = Field(default=70.0, description="Minimum knee angle for proper depth")
    max_threshold: float = Field(default=135.0, description="Maximum knee angle (standing)")
    severity: int = 8
    description: str = "Monitors knee flexion for proper squat depth"


class KneeValgusConfig(RuleConfig):
    """Knee valgus (inward collapse) configuration.
    
    Research: Hewett et al. (2005) - valgus angles > 10° increase
    ACL injury risk significantly.
    """
    
    max_threshold: float = Field(default=10.0, description="Maximum allowed valgus angle")
    severity: int = 9
    description: str = "Detects dangerous knee inward collapse"


class TorsoInclinationConfig(RuleConfig):
    """Torso inclination configuration.
    
    Research: Escamilla (2001) - forward lean between 30° and 75°
    is biomechanically safe for the spine.
    """
    
    min_threshold: float = Field(default=30.0, description="Minimum forward lean")
    max_threshold: float = Field(default=75.0, description="Maximum forward lean before unsafe")
    severity: int = 7
    description: str = "Monitors torso angle to protect spine"


class HipHingeConfig(RuleConfig):
    """Hip hinge mechanics configuration.
    
    Research: Hartmann et al. (2013) - proper hip flexion
    ranges from 45° to 100° during squat.
    """
    
    min_threshold: float = Field(default=45.0, description="Minimum hip angle")
    max_threshold: float = Field(default=100.0, description="Maximum hip angle")
    severity: int = 6
    description: str = "Ensures proper hip hinge mechanics"


class KneeOverToeConfig(RuleConfig):
    """Knee-over-toe positioning configuration.
    
    Research: Fry et al. (2003) - excessive knee forward travel
    increases patellofemoral stress.
    """
    
    max_threshold: float = Field(
        default=0.15,
        description="Maximum knee extension past toes as percentage of foot length"
    )
    severity: int = 5
    description: str = "Monitors safe knee positioning relative to toes"


class DepthAnalysisConfig(RuleConfig):
    """Squat depth analysis configuration.
    
    Research: NSCA Guidelines - thigh parallel or below
    for full range of motion benefits.
    """
    
    min_threshold: float = Field(
        default=0.0,
        description="Minimum angle below parallel (negative = ATG)"
    )
    max_threshold: float = Field(
        default=15.0,
        description="Maximum acceptable angle above parallel"
    )
    severity: int = 5
    description: str = "Verifies proper squat depth"


class RulesConfig(BaseModel):
    """Collection of all biomechanical rule configurations."""
    
    knee_flexion: KneeFlexionConfig = Field(default_factory=KneeFlexionConfig)
    knee_valgus: KneeValgusConfig = Field(default_factory=KneeValgusConfig)
    torso_inclination: TorsoInclinationConfig = Field(default_factory=TorsoInclinationConfig)
    hip_hinge: HipHingeConfig = Field(default_factory=HipHingeConfig)
    knee_over_toe: KneeOverToeConfig = Field(default_factory=KneeOverToeConfig)
    depth_analysis: DepthAnalysisConfig = Field(default_factory=DepthAnalysisConfig)


class FilterConfig(BaseModel):
    """One-Euro Filter configuration for signal smoothing.
    
    The One-Euro Filter is an adaptive low-pass filter that reduces
    jitter at low speeds while maintaining responsiveness at high speeds.
    """
    
    enabled: bool = True
    min_cutoff: float = Field(
        default=1.0,
        ge=0.001,
        description="Minimum cutoff frequency (lower = smoother)"
    )
    beta: float = Field(
        default=0.007,
        ge=0.0,
        description="Speed coefficient (higher = more responsive)"
    )
    d_cutoff: float = Field(
        default=1.0,
        ge=0.001,
        description="Derivative cutoff frequency"
    )


class PoseEstimationConfig(BaseModel):
    """Pose estimation model configuration."""
    
    model: PoseModelType = PoseModelType.YOLOV8N
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence"
    )
    device: str = Field(
        default="auto",
        description="Device for inference (auto, cpu, cuda, mps)"
    )
    half_precision: bool = Field(
        default=False,
        description="Use FP16 inference (GPU only)"
    )


class VisualizationConfig(BaseModel):
    """Visualization and overlay configuration."""
    
    show_skeleton: bool = True
    show_keypoints: bool = True
    show_angles: bool = True
    show_feedback: bool = True
    show_metrics: bool = True
    show_phase: bool = True
    
    skeleton_color: tuple[int, int, int] = (0, 255, 0)
    keypoint_color: tuple[int, int, int] = (255, 0, 0)
    warning_color: tuple[int, int, int] = (0, 165, 255)
    error_color: tuple[int, int, int] = (0, 0, 255)
    success_color: tuple[int, int, int] = (0, 255, 0)
    
    line_thickness: int = Field(default=2, ge=1, le=10)
    keypoint_radius: int = Field(default=5, ge=1, le=20)
    font_scale: float = Field(default=0.6, ge=0.1, le=3.0)


class CameraConfig(BaseModel):
    """Camera/video source configuration."""
    
    source: int | str = 0
    width: int = Field(default=1280, ge=320, le=4096)
    height: int = Field(default=720, ge=240, le=2160)
    fps: int = Field(default=30, ge=1, le=120)
    buffer_size: int = Field(default=1, ge=1, le=10)


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""
    
    target_fps: int = Field(default=30, ge=1, le=120)
    skip_frames: int = Field(default=0, ge=0, le=10)
    async_processing: bool = False
    gpu_acceleration: bool = True


class Settings(BaseSettings):
    """
    Main settings class with full validation and environment variable support.
    
    Settings can be loaded from:
        1. Environment variables (SQUAT_ANALYZER__FIELD)
        2. YAML configuration file
        3. Default values
    
    Example:
        >>> settings = Settings()
        >>> settings = Settings(_env_file=".env")
        >>> settings = Settings.from_yaml("config/settings.yaml")
    """
    
    model_config = SettingsConfigDict(
        env_prefix="SQUAT_ANALYZER__",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application settings
    app_name: str = "Squat Analyzer"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    mode: AnalysisMode = AnalysisMode.REALTIME
    
    # Sub-configurations
    pose: PoseEstimationConfig = Field(default_factory=PoseEstimationConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Paths
    model_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "squat_analyzer" / "models"
    )
    log_dir: Path = Field(default=Path("logs"))
    
    @field_validator("model_cache_dir", "log_dir", mode="before")
    @classmethod
    def parse_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file.
        
        Args:
            path: Path to YAML configuration file.
            
        Returns:
            Settings instance with values from YAML.
            
        Example:
            >>> settings = Settings.from_yaml("config/settings.yaml")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        
        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Export settings to a YAML file.
        
        Args:
            path: Destination path for YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def get_active_rules(self) -> dict[str, RuleConfig]:
        """Get all enabled rule configurations.
        
        Returns:
            Dictionary of rule name -> config for enabled rules.
        """
        rules_dict = self.rules.model_dump()
        return {
            name: getattr(self.rules, name)
            for name, config in rules_dict.items()
            if config.get("enabled", True)
        }
