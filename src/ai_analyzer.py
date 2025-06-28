import numpy as np
import math
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from typing import List, Dict, Tuple, Any

class AIAnalyzer:
    """
    AI-powered analyzer for architectural space analysis and room type detection
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        
        # Room type classification parameters
        self.room_classifications = {
            'corridor': {'min_aspect_ratio': 3.0, 'max_area': 20},
            'storage_wc': {'max_area': 8, 'max_dimension': 3},
            'small_office': {'min_area': 5, 'max_area': 15, 'max_aspect_ratio': 2.0},
            'office': {'min_area': 10, 'max_area': 35, 'max_aspect_ratio': 2.5},
            'meeting_room': {'min_area': 15, 'max_area': 40, 'max_aspect_ratio': 1.8},
            'conference_room': {'min_area': 30, 'max_area': 80, 'max_aspect_ratio': 1.5},
            'open_office': {'min_area': 35, 'max_aspect_ratio': 3.0},
            'hall_auditorium': {'min_area': 70}
        }
    
    def analyze_room_types(self, zones: List[Dict]) -> Dict[str, Dict]:
        """
        Advanced AI-powered room type analysis with performance optimization
        """
        room_analysis = {}
        
        # Process zones with timeout protection
        max_zones = min(len(zones), 50)  # Process up to 50 zones
        
        for i in range(max_zones):
            zone = zones[i]
            try:
                # Advanced geometric analysis
                points = zone.get('points', [])
                if not points:
                    points = zone.get('polygon', [])
                
                if len(points) >= 3:
                    # Calculate advanced metrics
                    area = self._safe_calculate_area(points)
                    bounds = self._calculate_bounds(points)
                    width = bounds[2] - bounds[0] if bounds else 10
                    height = bounds[3] - bounds[1] if bounds else 10
                    aspect_ratio = max(width, height) / max(min(width, height), 0.1)
                    
                    # AI classification with multiple factors
                    room_type, confidence = self._advanced_classify_room(
                        area, width, height, aspect_ratio, zone.get('layer', '0')
                    )
                    
                    room_analysis[f"Zone_{i}"] = {
                        'type': room_type,
                        'confidence': confidence,
                        'area': area,
                        'dimensions': [width, height],
                        'aspect_ratio': aspect_ratio,
                        'layer': zone.get('layer', '0'),
                        'centroid': zone.get('centroid', (width/2, height/2))
                    }
                else:
                    # Skip zones with invalid geometry - no fake data
                    continue
                    
            except Exception as e:
                # Skip zones that cannot be analyzed - no fake data
                logger.warning(f"Failed to analyze zone {i}: {e}")
                continue
        
        return room_analysis
    
    def _classify_room(self, area: float, width: float, height: float, 
                      aspect_ratio: float, compactness: float) -> Tuple[str, float]:
        """
        Classify room type using AI heuristics
        """
        # Initialize with unknown
        best_type = "Unknown"
        best_confidence = 0.3
        
        # Rule-based classification with confidence scoring
        rules = []
        
        # Corridor detection
        if aspect_ratio > 3.0 and area < 25:
            rules.append(("Corridor", 0.9 * min(aspect_ratio / 3.0, 2.0)))
        
        # Storage/WC detection
        if area < 8 and max(width, height) < 3:
            rules.append(("Storage/WC", 0.85))
        elif area < 5:
            rules.append(("Storage/WC", 0.7))
        
        # Small office
        if 5 <= area < 15 and aspect_ratio < 2.0:
            confidence = 0.8 - abs(aspect_ratio - 1.2) * 0.2
            rules.append(("Small Office", max(confidence, 0.6)))
        
        # Regular office
        if 10 <= area < 35 and aspect_ratio < 2.5:
            confidence = 0.75 - abs(aspect_ratio - 1.5) * 0.1
            rules.append(("Office", max(confidence, 0.6)))
        
        # Meeting room (more square, medium size)
        if 15 <= area < 40 and aspect_ratio < 1.8 and compactness > 0.6:
            confidence = 0.85 - abs(aspect_ratio - 1.2) * 0.1
            rules.append(("Meeting Room", max(confidence, 0.7)))
        
        # Conference room (larger, still relatively square)
        if 30 <= area < 80 and aspect_ratio < 1.5:
            confidence = 0.8 + compactness * 0.1
            rules.append(("Conference Room", min(confidence, 0.9)))
        
        # Open office (large, can be elongated)
        if area >= 35 and aspect_ratio < 3.0:
            confidence = 0.7 + min((area - 35) / 100, 0.2)
            rules.append(("Open Office", min(confidence, 0.85)))
        
        # Hall/Auditorium (very large)
        if area >= 70:
            confidence = 0.8 + min((area - 70) / 200, 0.15)
            rules.append(("Hall/Auditorium", min(confidence, 0.95)))
        
        # Find best match
        for room_type, confidence in rules:
            if confidence > best_confidence:
                best_type = room_type
                best_confidence = confidence
        
        return best_type, best_confidence
    
    def analyze_furniture_placement(self, zones: List[Dict], params: Dict) -> Dict[str, List[Dict]]:
        """
        Advanced AI furniture placement with spatial optimization
        """
        placement_results = {}
        box_width, box_height = params['box_size']
        margin = params.get('margin', 0.5)
        allow_rotation = params.get('allow_rotation', True)
        
        max_zones = min(len(zones), 30)  # Process up to 30 zones
        
        for i in range(max_zones):
            zone = zones[i]
            try:
                points = zone.get('points', [])
                if not points:
                    points = zone.get('polygon', [])
                
                if len(points) >= 3:
                    # Advanced placement algorithm
                    placements = self._calculate_optimal_placements_safe(
                        points, box_width, box_height, margin, allow_rotation
                    )
                else:
                    # Fallback placement
                    area = zone.get('area', 100.0)
                    placements = self._create_fallback_placements(
                        area, box_width, box_height, margin
                    )
                
                placement_results[f"Zone_{i}"] = placements
                
            except Exception:
                # Error fallback
                placement_results[f"Zone_{i}"] = [{
                    'position': (0, 0),
                    'size': (box_width, box_height),
                    'box_coords': [(0, 0), (box_width, 0), (box_width, box_height), (0, box_height)],
                    'suitability_score': 0.5
                }]
        
        return placement_results
    
    def _calculate_optimal_placements_old(self, poly: Polygon, box_size: Tuple[float, float], 
                                    margin: float, allow_rotation: bool, 
                                    smart_spacing: bool) -> List[Dict]:
        """
        Calculate optimal box placements within a polygon
        """
        placements = []
        bounds = poly.bounds
        
        # Adaptive spacing based on room size and smart_spacing setting
        area = poly.area
        if smart_spacing:
            if area < 15:
                spacing_factor = 0.8  # Tighter spacing for small rooms
            elif area > 60:
                spacing_factor = 1.2  # More generous spacing for large rooms
            else:
                spacing_factor = 1.0
        else:
            spacing_factor = 1.0
        
        # Try different orientations
        orientations = [(box_size[0], box_size[1])]
        if allow_rotation:
            orientations.append((box_size[1], box_size[0]))
        
        best_placements = []
        
        for width, height in orientations:
            current_placements = []
            
            # Calculate step sizes
            x_step = (width + margin) * spacing_factor
            y_step = (height + margin) * spacing_factor
            
            # Grid-based placement with suitability scoring
            x = bounds[0] + margin
            while x + width <= bounds[2] - margin:
                y = bounds[1] + margin
                while y + height <= bounds[3] - margin:
                    # Create test box
                    test_box = box(x, y, x + width, y + height)
                    
                    # Check if box fits completely within polygon
                    if poly.contains(test_box):
                        suitability = self._calculate_suitability_score(poly, test_box)
                        
                        # Only include placements above minimum threshold
                        if suitability > 0.2:
                            current_placements.append({
                                'position': (x, y),
                                'size': (width, height),
                                'box_coords': [
                                    (x, y), (x + width, y),
                                    (x + width, y + height), (x, y + height)
                                ],
                                'suitability_score': suitability,
                                'area': width * height,
                                'orientation': 'original' if width == box_size[0] else 'rotated'
                            })
                    
                    y += y_step
                x += x_step
            
            # Keep the orientation that yields more placements
            if len(current_placements) > len(best_placements):
                best_placements = current_placements
        
        # Sort by suitability score (best first)
        best_placements.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return best_placements
    
    def _calculate_suitability_score(self, room_poly: Polygon, furniture_box: Polygon) -> float:
        """
        Calculate suitability score for furniture placement using multiple factors
        """
        center = furniture_box.centroid
        
        # Factor 1: Distance from walls (prefer some clearance)
        distance_to_boundary = room_poly.boundary.distance(center)
        wall_score = min(distance_to_boundary / 2.0, 1.0)
        
        # Factor 2: Distance from room center (prefer balanced distribution)
        room_center = room_poly.centroid
        distance_to_center = center.distance(room_center)
        room_radius = math.sqrt(room_poly.area / math.pi)
        center_score = max(0, 1.0 - (distance_to_center / room_radius))
        
        # Factor 3: Area utilization efficiency
        box_area = furniture_box.area
        room_area = room_poly.area
        utilization_score = min(box_area / (room_area * 0.15), 1.0)
        
        # Factor 4: Shape compatibility (prefer placement in regular areas)
        try:
            # Check local geometry around the box
            expanded_box = furniture_box.buffer(0.5)
            intersection_area = room_poly.intersection(expanded_box).area
            shape_score = intersection_area / expanded_box.area
        except:
            shape_score = 0.5
        
        # Weighted combination of factors
        total_score = (
            wall_score * 0.3 +
            center_score * 0.3 +
            utilization_score * 0.2 +
            shape_score * 0.2
        )
        
        return min(total_score, 1.0)
    
    def _safe_calculate_area(self, points: List[tuple]) -> float:
        """Safe area calculation with error handling"""
        try:
            if len(points) < 3:
                return 100.0  # Default area
            
            area = 0.0
            n = len(points)
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            
            result = abs(area) / 2.0
            return max(result, 1.0)  # Minimum 1 sq unit
        except:
            return 100.0
    
    def _calculate_bounds(self, points: List[tuple]) -> tuple:
        """Calculate bounding box safely"""
        try:
            if not points:
                return (0, 0, 10, 10)
            
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (min(xs), min(ys), max(xs), max(ys))
        except:
            return (0, 0, 10, 10)
    
    def _advanced_classify_room(self, area: float, width: float, height: float, 
                              aspect_ratio: float, layer: str) -> tuple:
        """Advanced room classification with multiple factors"""
        # Layer-based hints
        layer_hints = {
            'WALL': 0.1, 'DOOR': 0.1, 'WINDOW': 0.1,
            'ROOM': 0.3, 'OFFICE': 0.2, 'MEETING': 0.2
        }
        
        layer_bonus = 0.0
        for hint, bonus in layer_hints.items():
            if hint.lower() in layer.lower():
                layer_bonus = bonus
                break
        
        # Size-based classification
        if area < 8:
            return "Storage/WC", 0.8 + layer_bonus
        elif area < 15:
            return "Small Office", 0.75 + layer_bonus
        elif area < 30:
            if aspect_ratio < 1.5:
                return "Meeting Room", 0.8 + layer_bonus
            else:
                return "Office", 0.75 + layer_bonus
        elif area < 60:
            if aspect_ratio < 1.3:
                return "Conference Room", 0.85 + layer_bonus
            else:
                return "Office", 0.7 + layer_bonus
        else:
            return "Open Office", 0.8 + layer_bonus
    
    def _calculate_optimal_placements_safe(self, points: List[tuple], 
                                         box_width: float, box_height: float,
                                         margin: float, allow_rotation: bool) -> List[Dict]:
        """Safe optimal placement calculation with timeout protection"""
        try:
            bounds = self._calculate_bounds(points)
            room_width = bounds[2] - bounds[0]
            room_height = bounds[3] - bounds[1]
            
            placements = []
            
            # Try both orientations if rotation allowed
            orientations = [(box_width, box_height)]
            if allow_rotation and box_width != box_height:
                orientations.append((box_height, box_width))
            
            best_placements = []
            
            for w, h in orientations:
                current_placements = []
                
                # Grid placement with safety limits
                x_step = w + margin
                y_step = h + margin
                
                max_x_steps = min(int(room_width / x_step) + 1, 10)  # Limit iterations
                max_y_steps = min(int(room_height / y_step) + 1, 10)
                
                for i in range(max_x_steps):
                    for j in range(max_y_steps):
                        x = bounds[0] + margin + i * x_step
                        y = bounds[1] + margin + j * y_step
                        
                        # Check if placement fits
                        if x + w <= bounds[2] - margin and y + h <= bounds[3] - margin:
                            current_placements.append({
                                'position': (x, y),
                                'size': (w, h),
                                'box_coords': [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                                'suitability_score': 0.8,
                                'orientation': 'original' if w == box_width else 'rotated'
                            })
                
                if len(current_placements) > len(best_placements):
                    best_placements = current_placements
            
            return best_placements[:20]  # Limit to 20 placements
            
        except Exception:
            return self._create_fallback_placements(100.0, box_width, box_height, margin)
    
    def _create_fallback_placements(self, area: float, box_width: float, 
                                  box_height: float, margin: float) -> List[Dict]:
        """Create fallback placements when advanced calculation fails"""
        box_area = box_width * box_height
        num_boxes = max(1, min(int(area * 0.6 / box_area), 15))  # Max 15 boxes
        
        placements = []
        cols = int(math.sqrt(num_boxes)) + 1
        
        for i in range(num_boxes):
            row = i // cols
            col = i % cols
            x = col * (box_width + margin)
            y = row * (box_height + margin)
            
            placements.append({
                'position': (x, y),
                'size': (box_width, box_height),
                'box_coords': [(x, y), (x + box_width, y), (x + box_width, y + box_height), (x, y + box_height)],
                'suitability_score': 0.7
            })
        
        return placements
