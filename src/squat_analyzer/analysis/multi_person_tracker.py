"""
Multi-Person Squat Tracker
==========================

Tracks MULTIPLE people simultaneously, each in their own horizontal zone.
Maintains separate state for each person to enable per-person rep counting.

Architecture:
    1. On first detection, create a zone for EACH person
    2. Each subsequent frame, assign detections to the nearest zone
    3. Return all tracked people with their track IDs
    4. Server maintains separate SquatDetector per track ID
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque

from squat_analyzer.core.keypoints import Keypoints
from squat_analyzer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PersonTrack:
    """Individual person track with zone locking."""
    track_id: int
    
    # Horizontal zone (locked based on initial position)
    zone_center: float
    zone_min: float
    zone_max: float
    
    # Position state
    last_x: float
    last_y: float
    
    # Tracking state
    frames_tracked: int = 0
    frames_lost: int = 0
    
    # Position history for smoothing
    x_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def is_in_zone(self, x: float) -> bool:
        """Check if x position is within this track's zone."""
        return self.zone_min <= x <= self.zone_max
    
    def distance_to_zone_center(self, x: float) -> float:
        """Distance from x to zone center."""
        return abs(x - self.zone_center)
    
    def update(self, x: float, y: float):
        """Update track with new position."""
        self.last_x = x
        self.last_y = y
        self.x_history.append(x)
        self.frames_tracked += 1
        self.frames_lost = 0
    
    def mark_lost(self):
        """Mark track as lost for this frame."""
        self.frames_lost += 1


class MultiPersonTracker:
    """
    Tracks multiple people using horizontal zone locking.
    
    Each person gets their own zone based on their initial horizontal position.
    This allows tracking multiple people doing synchronized movements.
    """
    
    def __init__(
        self,
        zone_width_factor: float = 0.8,   # Zone width as factor of person width
        zone_margin: float = 0.05,         # Additional margin (fraction of frame)
        max_lost_frames: int = 30,         # Frames before removing track
        min_separation: float = 0.1,       # Minimum separation between zones (fraction of frame)
    ):
        self.zone_width_factor = zone_width_factor
        self.zone_margin = zone_margin
        self.max_lost_frames = max_lost_frames
        self.min_separation = min_separation
        
        # Active tracks
        self._tracks: Dict[int, PersonTrack] = {}
        self._next_track_id: int = 1
        self._initialized: bool = False
        self._frame_width: float = 1920
    
    def _get_person_x(self, kp: Keypoints) -> Optional[float]:
        """Get horizontal center of person."""
        try:
            return (kp.left_hip[0] + kp.right_hip[0] + 
                    kp.left_shoulder[0] + kp.right_shoulder[0]) / 4
        except:
            return None
    
    def _get_person_y(self, kp: Keypoints) -> Optional[float]:
        """Get vertical center of person (hip level)."""
        try:
            return (kp.left_hip[1] + kp.right_hip[1]) / 2
        except:
            return None
    
    def _get_person_width(self, kp: Keypoints) -> float:
        """Get approximate width of person."""
        try:
            return max(
                abs(kp.right_shoulder[0] - kp.left_shoulder[0]),
                abs(kp.right_hip[0] - kp.left_hip[0])
            )
        except:
            return 100
    
    def _create_zone(self, x: float, width: float) -> Tuple[float, float, float]:
        """Create zone boundaries for a person."""
        # Zone width based on person width + margin
        half_zone = (width * self.zone_width_factor / 2) + (self._frame_width * self.zone_margin)
        
        zone_min = max(0, x - half_zone)
        zone_max = min(self._frame_width, x + half_zone)
        
        return x, zone_min, zone_max
    
    def _create_track(self, kp: Keypoints) -> PersonTrack:
        """Create a new track for a person."""
        x = self._get_person_x(kp)
        y = self._get_person_y(kp)
        width = self._get_person_width(kp)
        
        zone_center, zone_min, zone_max = self._create_zone(x, width)
        
        track = PersonTrack(
            track_id=self._next_track_id,
            zone_center=zone_center,
            zone_min=zone_min,
            zone_max=zone_max,
            last_x=x,
            last_y=y,
        )
        
        self._next_track_id += 1
        
        logger.info(f"Track #{track.track_id} created: x={x:.0f}, zone=[{zone_min:.0f}, {zone_max:.0f}]")
        
        return track
    
    def _find_best_track(self, x: float) -> Optional[PersonTrack]:
        """Find the track whose zone contains this x position."""
        candidates = []
        
        for track in self._tracks.values():
            if track.is_in_zone(x):
                dist = track.distance_to_zone_center(x)
                candidates.append((dist, track))
        
        if not candidates:
            return None
        
        # Return track with closest zone center
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]
    
    def _zones_overlap(self, zone1: Tuple[float, float], zone2: Tuple[float, float]) -> bool:
        """Check if two zones overlap significantly."""
        min1, max1 = zone1
        min2, max2 = zone2
        
        # Calculate overlap
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        
        if overlap_start >= overlap_end:
            return False
        
        # Check if overlap is significant (more than 50% of either zone)
        overlap_size = overlap_end - overlap_start
        zone1_size = max1 - min1
        zone2_size = max2 - min2
        
        return overlap_size > 0.5 * min(zone1_size, zone2_size)
    
    def select_persons(
        self,
        keypoints_list: List[Keypoints],
        frame_width: int,
        frame_height: int,
    ) -> Dict[int, Keypoints]:
        """
        Select and track all persons in frame.
        
        Returns:
            Dictionary mapping track_id -> Keypoints for each tracked person
        """
        self._frame_width = frame_width
        
        # No detections
        if not keypoints_list:
            # Mark all tracks as lost
            for track in self._tracks.values():
                track.mark_lost()
            
            # Remove tracks that have been lost too long
            self._tracks = {
                tid: track for tid, track in self._tracks.items()
                if track.frames_lost < self.max_lost_frames
            }
            
            return {}
        
        # First frame - initialize all tracks
        if not self._initialized:
            return self._initialize_tracks(keypoints_list)
        
        # Match detections to existing tracks
        return self._match_detections(keypoints_list)
    
    def _initialize_tracks(self, keypoints_list: List[Keypoints]) -> Dict[int, Keypoints]:
        """Initialize tracks for all detected persons."""
        result = {}
        
        # Sort persons by x position (left to right)
        persons = []
        for kp in keypoints_list:
            x = self._get_person_x(kp)
            if x is not None:
                persons.append((x, kp))
        
        persons.sort(key=lambda p: p[0])
        
        # Create tracks, ensuring non-overlapping zones
        created_zones = []
        
        for x, kp in persons:
            width = self._get_person_width(kp)
            zone_center, zone_min, zone_max = self._create_zone(x, width)
            
            # Check if this zone overlaps with existing zones
            overlaps = False
            for existing_zone in created_zones:
                if self._zones_overlap((zone_min, zone_max), existing_zone):
                    overlaps = True
                    break
            
            if not overlaps:
                track = self._create_track(kp)
                self._tracks[track.track_id] = track
                created_zones.append((zone_min, zone_max))
                result[track.track_id] = kp
        
        if result:
            self._initialized = True
            logger.info(f"Initialized {len(result)} tracks")
        
        return result
    
    def _match_detections(self, keypoints_list: List[Keypoints]) -> Dict[int, Keypoints]:
        """Match detections to existing tracks, and create new tracks for new people."""
        result = {}
        matched_tracks = set()
        unmatched_detections = []
        
        # For each detection, find best matching track
        for kp in keypoints_list:
            x = self._get_person_x(kp)
            y = self._get_person_y(kp)
            
            if x is None:
                continue
            
            track = self._find_best_track(x)
            
            if track is not None and track.track_id not in matched_tracks:
                track.update(x, y)
                matched_tracks.add(track.track_id)
                result[track.track_id] = kp
            elif track is None:
                # Detection doesn't match any track - potential new person
                unmatched_detections.append(kp)
        
        # Try to create new tracks for unmatched detections
        for kp in unmatched_detections:
            x = self._get_person_x(kp)
            if x is None:
                continue
            
            width = self._get_person_width(kp)
            zone_center, zone_min, zone_max = self._create_zone(x, width)
            
            # Check if this zone overlaps with existing track zones
            overlaps = False
            for track in self._tracks.values():
                if self._zones_overlap((zone_min, zone_max), (track.zone_min, track.zone_max)):
                    overlaps = True
                    break
            
            if not overlaps:
                track = self._create_track(kp)
                self._tracks[track.track_id] = track
                result[track.track_id] = kp
        
        # Mark unmatched tracks as lost
        for track_id, track in self._tracks.items():
            if track_id not in matched_tracks:
                track.mark_lost()
        
        # Remove tracks that have been lost too long
        self._tracks = {
            tid: track for tid, track in self._tracks.items()
            if track.frames_lost < self.max_lost_frames
        }
        
        return result
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self._tracks)
    
    def reset(self):
        """Reset tracker for new session."""
        self._tracks.clear()
        self._next_track_id = 1
        self._initialized = False
        logger.info("MultiPersonTracker reset")
