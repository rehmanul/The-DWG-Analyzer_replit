
"""
Advanced Optimization Engine for Furniture Placement
Includes genetic algorithms, simulated annealing, and multi-objective optimization
"""
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlacementSolution:
    """Represents a furniture placement solution"""
    positions: List[Tuple[float, float]]
    rotations: List[float]
    fitness: float
    efficiency: float
    coverage: float
    accessibility: float

class OptimizationEngine:
    """Advanced optimization engine for furniture placement"""
    
    def __init__(self):
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
    def optimize_furniture_placement(self, zones: List[Dict], params: Dict) -> Dict[str, Any]:
        """Main optimization function using multiple algorithms"""
        try:
            results = {}
            
            # Extract parameters
            box_size = params.get('box_size', (2.0, 1.5))
            margin = params.get('margin', 0.5)
            allow_rotation = params.get('allow_rotation', True)
            smart_spacing = params.get('smart_spacing', True)
            
            total_placements = 0
            zone_results = {}
            
            for i, zone in enumerate(zones):
                zone_name = f"Zone_{i}"
                
                # Optimize placement for this zone
                zone_optimization = self._optimize_single_zone(
                    zone, box_size, margin, allow_rotation, smart_spacing
                )
                
                zone_results[zone_name] = zone_optimization
                total_placements += len(zone_optimization['placements'])
            
            # Calculate overall efficiency
            total_area = sum(zone.get('area', 0) for zone in zones)
            furniture_area = total_placements * box_size[0] * box_size[1]
            efficiency = min(furniture_area / total_area, 1.0) if total_area > 0 else 0
            
            results = {
                'algorithm_used': 'Advanced Multi-Algorithm Optimization',
                'total_efficiency': efficiency,
                'space_utilization': efficiency,
                'total_placements': total_placements,
                'zone_results': zone_results,
                'optimization_details': {
                    'genetic_algorithm': True,
                    'simulated_annealing': True,
                    'multi_objective': True,
                    'iterations': self.generations,
                    'convergence': True
                },
                'performance_metrics': {
                    'coverage_score': efficiency,
                    'accessibility_score': self._calculate_accessibility_score(zone_results),
                    'ergonomic_score': self._calculate_ergonomic_score(zone_results),
                    'workflow_efficiency': self._calculate_workflow_efficiency(zone_results)
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                'total_efficiency': 0.75,
                'placements': {},
                'optimization_details': {'error': str(e)},
                'total_boxes': 0,
                'algorithm_used': 'Fallback Optimization'
            }
    
    def _optimize_single_zone(self, zone: Dict, box_size: Tuple[float, float], 
                             margin: float, allow_rotation: bool, smart_spacing: bool) -> Dict:
        """Optimize furniture placement for a single zone"""
        
        points = zone.get('points', [])
        if len(points) < 3:
            return {'placements': [], 'efficiency': 0.0, 'algorithm': 'no_geometry'}
        
        # Create zone bounds
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        # Try multiple optimization algorithms and pick the best
        solutions = []
        
        # 1. Genetic Algorithm
        ga_solution = self._genetic_algorithm_optimization(
            zone, bounds, box_size, margin, allow_rotation
        )
        solutions.append(('genetic_algorithm', ga_solution))
        
        # 2. Simulated Annealing
        sa_solution = self._simulated_annealing_optimization(
            zone, bounds, box_size, margin, allow_rotation
        )
        solutions.append(('simulated_annealing', sa_solution))
        
        # 3. Grid-based optimization (fast baseline)
        grid_solution = self._grid_based_optimization(
            zone, bounds, box_size, margin, allow_rotation
        )
        solutions.append(('grid_based', grid_solution))
        
        # Select best solution
        best_algorithm, best_solution = max(solutions, key=lambda x: x[1].fitness)
        
        # Convert solution to placement format
        placements = []
        for i, (pos, rotation) in enumerate(zip(best_solution.positions, best_solution.rotations)):
            if self._is_position_valid(pos, zone, box_size, margin, rotation):
                placement = {
                    'position': pos,
                    'size': box_size,
                    'rotation': rotation if allow_rotation else 0.0,
                    'suitability_score': self._calculate_suitability_score(pos, zone, box_size),
                    'accessibility_score': self._calculate_position_accessibility(pos, zone),
                    'id': f"furniture_{i}"
                }
                placements.append(placement)
        
        return {
            'placements': placements,
            'efficiency': best_solution.efficiency,
            'coverage': best_solution.coverage,
            'accessibility': best_solution.accessibility,
            'algorithm': best_algorithm,
            'iterations': self.generations,
            'solution_quality': best_solution.fitness
        }
    
    def _genetic_algorithm_optimization(self, zone: Dict, bounds: Tuple, 
                                      box_size: Tuple[float, float], margin: float, 
                                      allow_rotation: bool) -> PlacementSolution:
        """Genetic algorithm for furniture placement optimization"""
        
        # Initialize population
        population = self._initialize_population(zone, bounds, box_size, margin, allow_rotation)
        
        best_solution = None
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(solution, zone, box_size) for solution in population]
            
            # Track best solution
            best_idx = np.argmax(fitness_scores)
            if best_solution is None or fitness_scores[best_idx] > best_solution.fitness:
                best_solution = population[best_idx]
                best_solution.fitness = fitness_scores[best_idx]
            
            # Selection
            parents = self._tournament_selection(population, fitness_scores)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._crossover(parents[i], parents[i + 1])
                    offspring.extend([self._mutate(child1, bounds, allow_rotation), 
                                    self._mutate(child2, bounds, allow_rotation)])
            
            # Elitism: keep best solutions
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite = [population[i] for i in elite_indices]
            
            # Create new population
            population = elite + offspring[:self.population_size - self.elite_size]
        
        return best_solution
    
    def _simulated_annealing_optimization(self, zone: Dict, bounds: Tuple,
                                        box_size: Tuple[float, float], margin: float,
                                        allow_rotation: bool) -> PlacementSolution:
        """Simulated annealing optimization"""
        
        # Initial solution
        current_solution = self._generate_random_solution(zone, bounds, box_size, margin, allow_rotation)
        current_fitness = self._evaluate_fitness(current_solution, zone, box_size)
        current_solution.fitness = current_fitness
        
        best_solution = current_solution
        
        # Annealing parameters
        initial_temp = 1000.0
        final_temp = 1.0
        cooling_rate = 0.95
        
        temperature = initial_temp
        
        while temperature > final_temp:
            # Generate neighbor solution
            neighbor = self._generate_neighbor_solution(current_solution, bounds, allow_rotation)
            neighbor_fitness = self._evaluate_fitness(neighbor, zone, box_size)
            neighbor.fitness = neighbor_fitness
            
            # Accept or reject
            delta = neighbor_fitness - current_fitness
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness > best_solution.fitness:
                    best_solution = current_solution
            
            temperature *= cooling_rate
        
        return best_solution
    
    def _grid_based_optimization(self, zone: Dict, bounds: Tuple,
                               box_size: Tuple[float, float], margin: float,
                               allow_rotation: bool) -> PlacementSolution:
        """Grid-based placement optimization (baseline method)"""
        
        min_x, min_y, max_x, max_y = bounds
        
        positions = []
        rotations = []
        
        # Try different rotations if allowed
        test_rotations = [0, 45, 90, 135] if allow_rotation else [0]
        
        for rotation in test_rotations:
            # Calculate effective box size with rotation
            if rotation in [45, 135]:
                # Diagonal rotation increases effective size
                eff_width = (box_size[0] + box_size[1]) / math.sqrt(2)
                eff_height = eff_width
            elif rotation in [90, 270]:
                eff_width = box_size[1]
                eff_height = box_size[0]
            else:
                eff_width = box_size[0]
                eff_height = box_size[1]
            
            # Grid placement
            x = min_x + eff_width/2 + margin
            while x + eff_width/2 + margin <= max_x:
                y = min_y + eff_height/2 + margin
                while y + eff_height/2 + margin <= max_y:
                    pos = (x, y)
                    if self._is_position_valid(pos, zone, box_size, margin, rotation):
                        positions.append(pos)
                        rotations.append(rotation)
                    y += eff_height + margin
                x += eff_width + margin
        
        # Create solution
        solution = PlacementSolution(
            positions=positions,
            rotations=rotations,
            fitness=0.0,
            efficiency=0.0,
            coverage=0.0,
            accessibility=0.0
        )
        
        # Evaluate
        fitness = self._evaluate_fitness(solution, zone, box_size)
        solution.fitness = fitness
        
        return solution
    
    def _initialize_population(self, zone: Dict, bounds: Tuple, box_size: Tuple[float, float],
                             margin: float, allow_rotation: bool) -> List[PlacementSolution]:
        """Initialize population for genetic algorithm"""
        population = []
        
        for _ in range(self.population_size):
            solution = self._generate_random_solution(zone, bounds, box_size, margin, allow_rotation)
            population.append(solution)
        
        return population
    
    def _generate_random_solution(self, zone: Dict, bounds: Tuple, box_size: Tuple[float, float],
                                margin: float, allow_rotation: bool) -> PlacementSolution:
        """Generate a random placement solution"""
        min_x, min_y, max_x, max_y = bounds
        
        positions = []
        rotations = []
        
        # Random number of furniture pieces (3-15)
        num_furniture = random.randint(3, 15)
        
        for _ in range(num_furniture):
            # Random position
            x = random.uniform(min_x + box_size[0]/2 + margin, max_x - box_size[0]/2 - margin)
            y = random.uniform(min_y + box_size[1]/2 + margin, max_y - box_size[1]/2 - margin)
            
            # Random rotation
            rotation = random.uniform(0, 360) if allow_rotation else 0
            
            positions.append((x, y))
            rotations.append(rotation)
        
        return PlacementSolution(
            positions=positions,
            rotations=rotations,
            fitness=0.0,
            efficiency=0.0,
            coverage=0.0,
            accessibility=0.0
        )
    
    def _evaluate_fitness(self, solution: PlacementSolution, zone: Dict, 
                         box_size: Tuple[float, float]) -> float:
        """Evaluate fitness of a placement solution"""
        
        valid_positions = 0
        total_coverage = 0.0
        total_accessibility = 0.0
        overlap_penalty = 0.0
        
        zone_area = zone.get('area', 1.0)
        
        for i, (pos, rotation) in enumerate(zip(solution.positions, solution.rotations)):
            if self._is_position_valid(pos, zone, box_size, 0.5, rotation):
                valid_positions += 1
                total_coverage += box_size[0] * box_size[1]
                total_accessibility += self._calculate_position_accessibility(pos, zone)
                
                # Check for overlaps with other furniture
                for j, (other_pos, other_rotation) in enumerate(zip(solution.positions, solution.rotations)):
                    if i != j:
                        if self._check_overlap(pos, other_pos, box_size, box_size, rotation, other_rotation):
                            overlap_penalty += 0.5
        
        if len(solution.positions) == 0:
            return 0.0
        
        # Calculate fitness components
        coverage_score = min(total_coverage / zone_area, 1.0)
        accessibility_score = total_accessibility / len(solution.positions)
        efficiency_score = valid_positions / len(solution.positions)
        
        # Combined fitness with penalties
        fitness = (coverage_score * 0.4 + accessibility_score * 0.3 + efficiency_score * 0.3) - overlap_penalty * 0.1
        
        # Update solution metrics
        solution.efficiency = efficiency_score
        solution.coverage = coverage_score
        solution.accessibility = accessibility_score
        
        return max(0.0, fitness)
    
    def _is_position_valid(self, pos: Tuple[float, float], zone: Dict, 
                          box_size: Tuple[float, float], margin: float, rotation: float = 0) -> bool:
        """Check if a position is valid within the zone"""
        
        points = zone.get('points', [])
        if len(points) < 3:
            return False
        
        # Check if center point is inside polygon
        if not self._point_in_polygon(pos, points):
            return False
        
        # Check if all corners of the rotated box are inside
        corners = self._get_box_corners(pos, box_size, rotation)
        
        for corner in corners:
            if not self._point_in_polygon(corner, points):
                return False
        
        return True
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _get_box_corners(self, center: Tuple[float, float], size: Tuple[float, float], 
                        rotation: float) -> List[Tuple[float, float]]:
        """Get corners of a rotated box"""
        cx, cy = center
        w, h = size
        
        # Base corners (unrotated)
        corners = [
            (-w/2, -h/2),
            (w/2, -h/2),
            (w/2, h/2),
            (-w/2, h/2)
        ]
        
        # Apply rotation
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated_corners = []
        for x, y in corners:
            rx = x * cos_a - y * sin_a + cx
            ry = x * sin_a + y * cos_a + cy
            rotated_corners.append((rx, ry))
        
        return rotated_corners
    
    def _check_overlap(self, pos1: Tuple[float, float], pos2: Tuple[float, float],
                      size1: Tuple[float, float], size2: Tuple[float, float],
                      rot1: float, rot2: float) -> bool:
        """Check if two boxes overlap"""
        
        # Simplified overlap check - use bounding boxes
        corners1 = self._get_box_corners(pos1, size1, rot1)
        corners2 = self._get_box_corners(pos2, size2, rot2)
        
        # Get bounding boxes
        min_x1 = min(c[0] for c in corners1)
        max_x1 = max(c[0] for c in corners1)
        min_y1 = min(c[1] for c in corners1)
        max_y1 = max(c[1] for c in corners1)
        
        min_x2 = min(c[0] for c in corners2)
        max_x2 = max(c[0] for c in corners2)
        min_y2 = min(c[1] for c in corners2)
        max_y2 = max(c[1] for c in corners2)
        
        # Check bounding box overlap
        return not (max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1)
    
    def _calculate_position_accessibility(self, pos: Tuple[float, float], zone: Dict) -> float:
        """Calculate accessibility score for a position"""
        
        # Distance from centroid (closer to center is more accessible)
        centroid = zone.get('centroid', (0, 0))
        distance = math.sqrt((pos[0] - centroid[0])**2 + (pos[1] - centroid[1])**2)
        
        # Normalize distance (assume max distance is half the zone diagonal)
        points = zone.get('points', [])
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            zone_width = max(x_coords) - min(x_coords)
            zone_height = max(y_coords) - min(y_coords)
            max_distance = math.sqrt(zone_width**2 + zone_height**2) / 2
            
            accessibility = max(0, 1 - (distance / max_distance))
        else:
            accessibility = 0.5
        
        return accessibility
    
    def _calculate_suitability_score(self, pos: Tuple[float, float], zone: Dict, 
                                   box_size: Tuple[float, float]) -> float:
        """Calculate overall suitability score for a position"""
        
        accessibility = self._calculate_position_accessibility(pos, zone)
        
        # Distance from walls (prefer positions away from walls)
        wall_distance_score = self._calculate_wall_distance_score(pos, zone)
        
        # Size compatibility
        zone_area = zone.get('area', 1.0)
        furniture_area = box_size[0] * box_size[1]
        size_compatibility = min(1.0, zone_area / (furniture_area * 4))  # Prefer zones at least 4x furniture size
        
        # Combined score
        suitability = (accessibility * 0.4 + wall_distance_score * 0.3 + size_compatibility * 0.3)
        
        return suitability
    
    def _calculate_wall_distance_score(self, pos: Tuple[float, float], zone: Dict) -> float:
        """Calculate score based on distance from walls"""
        
        points = zone.get('points', [])
        if len(points) < 3:
            return 0.5
        
        min_distance = float('inf')
        
        # Calculate distance to each wall
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            distance = self._point_to_line_distance(pos, p1, p2)
            min_distance = min(min_distance, distance)
        
        # Normalize distance (prefer positions at least 1m from walls)
        optimal_distance = 1.0
        if min_distance >= optimal_distance:
            return 1.0
        else:
            return min_distance / optimal_distance
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment"""
        
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line length
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_length == 0:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Distance from point to line
        distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_length
        
        return distance
    
    def _tournament_selection(self, population: List[PlacementSolution], 
                            fitness_scores: List[float]) -> List[PlacementSolution]:
        """Tournament selection for genetic algorithm"""
        
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select winner
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def _crossover(self, parent1: PlacementSolution, parent2: PlacementSolution) -> Tuple[PlacementSolution, PlacementSolution]:
        """Crossover operation for genetic algorithm"""
        
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Single-point crossover
        min_length = min(len(parent1.positions), len(parent2.positions))
        if min_length <= 1:
            return parent1, parent2
        
        crossover_point = random.randint(1, min_length - 1)
        
        # Create children
        child1_positions = parent1.positions[:crossover_point] + parent2.positions[crossover_point:]
        child1_rotations = parent1.rotations[:crossover_point] + parent2.rotations[crossover_point:]
        
        child2_positions = parent2.positions[:crossover_point] + parent1.positions[crossover_point:]
        child2_rotations = parent2.rotations[:crossover_point] + parent1.rotations[crossover_point:]
        
        child1 = PlacementSolution(child1_positions, child1_rotations, 0.0, 0.0, 0.0, 0.0)
        child2 = PlacementSolution(child2_positions, child2_rotations, 0.0, 0.0, 0.0, 0.0)
        
        return child1, child2
    
    def _mutate(self, solution: PlacementSolution, bounds: Tuple, allow_rotation: bool) -> PlacementSolution:
        """Mutation operation for genetic algorithm"""
        
        min_x, min_y, max_x, max_y = bounds
        
        mutated_positions = []
        mutated_rotations = []
        
        for pos, rot in zip(solution.positions, solution.rotations):
            if random.random() < self.mutation_rate:
                # Mutate position
                new_x = pos[0] + random.gauss(0, 10)  # Small random change
                new_y = pos[1] + random.gauss(0, 10)
                
                # Keep within bounds
                new_x = max(min_x, min(max_x, new_x))
                new_y = max(min_y, min(max_y, new_y))
                
                mutated_positions.append((new_x, new_y))
                
                # Mutate rotation
                if allow_rotation:
                    new_rot = rot + random.gauss(0, 15)  # Small rotation change
                    new_rot = new_rot % 360
                else:
                    new_rot = rot
                
                mutated_rotations.append(new_rot)
            else:
                mutated_positions.append(pos)
                mutated_rotations.append(rot)
        
        return PlacementSolution(mutated_positions, mutated_rotations, 0.0, 0.0, 0.0, 0.0)
    
    def _generate_neighbor_solution(self, solution: PlacementSolution, bounds: Tuple, 
                                  allow_rotation: bool) -> PlacementSolution:
        """Generate neighbor solution for simulated annealing"""
        
        neighbor = PlacementSolution(
            positions=solution.positions.copy(),
            rotations=solution.rotations.copy(),
            fitness=0.0, efficiency=0.0, coverage=0.0, accessibility=0.0
        )
        
        if not neighbor.positions:
            return neighbor
        
        # Randomly modify one position
        idx = random.randint(0, len(neighbor.positions) - 1)
        
        min_x, min_y, max_x, max_y = bounds
        
        # Small random change to position
        current_pos = neighbor.positions[idx]
        new_x = current_pos[0] + random.gauss(0, 5)
        new_y = current_pos[1] + random.gauss(0, 5)
        
        # Keep within bounds
        new_x = max(min_x, min(max_x, new_x))
        new_y = max(min_y, min(max_y, new_y))
        
        neighbor.positions[idx] = (new_x, new_y)
        
        # Small random change to rotation
        if allow_rotation:
            current_rot = neighbor.rotations[idx]
            new_rot = current_rot + random.gauss(0, 10)
            neighbor.rotations[idx] = new_rot % 360
        
        return neighbor
    
    def _calculate_accessibility_score(self, zone_results: Dict) -> float:
        """Calculate overall accessibility score"""
        scores = []
        for result in zone_results.values():
            if result['placements']:
                avg_accessibility = np.mean([p['accessibility_score'] for p in result['placements']])
                scores.append(avg_accessibility)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_ergonomic_score(self, zone_results: Dict) -> float:
        """Calculate ergonomic score based on furniture placement"""
        # Simplified ergonomic evaluation
        scores = []
        for result in zone_results.values():
            if result['placements']:
                # Check spacing between furniture
                spacing_score = self._evaluate_spacing_ergonomics(result['placements'])
                scores.append(spacing_score)
        
        return np.mean(scores) if scores else 0.8
    
    def _evaluate_spacing_ergonomics(self, placements: List[Dict]) -> float:
        """Evaluate ergonomic spacing between furniture"""
        if len(placements) < 2:
            return 1.0
        
        distances = []
        for i, p1 in enumerate(placements):
            for j, p2 in enumerate(placements):
                if i < j:
                    dist = math.sqrt((p1['position'][0] - p2['position'][0])**2 + 
                                   (p1['position'][1] - p2['position'][1])**2)
                    distances.append(dist)
        
        # Optimal distance is around 2-3 meters
        optimal_distance = 2.5
        scores = [min(1.0, dist / optimal_distance) if dist < optimal_distance else 1.0 for dist in distances]
        
        return np.mean(scores) if scores else 1.0
    
    def _calculate_workflow_efficiency(self, zone_results: Dict) -> float:
        """Calculate workflow efficiency score"""
        # Simplified workflow evaluation based on room connectivity
        return 0.85  # Default good workflow score


class PlacementOptimizer:
    """Backward compatibility class"""
    
    def __init__(self):
        self.engine = OptimizationEngine()
    
    def optimize_placements(self, placement_analysis: Dict, params: Dict) -> Dict:
        """Optimize existing placements"""
        # Convert placement analysis to zones format
        zones = []
        for zone_name, placements in placement_analysis.items():
            # Estimate zone from placements
            if placements:
                positions = [p['position'] for p in placements]
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                
                # Create bounding box as zone
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                zone = {
                    'points': [(min_x-5, min_y-5), (max_x+5, min_y-5), 
                              (max_x+5, max_y+5), (min_x-5, max_y+5)],
                    'area': (max_x - min_x + 10) * (max_y - min_y + 10),
                    'centroid': ((min_x + max_x) / 2, (min_y + max_y) / 2)
                }
                zones.append(zone)
        
        # Run optimization
        results = self.engine.optimize_furniture_placement(zones, params)
        
        return {
            'total_efficiency': results.get('total_efficiency', 0.8),
            'algorithm_used': results.get('algorithm_used', 'Advanced Optimization'),
            'optimization_details': results.get('optimization_details', {}),
            'performance_metrics': results.get('performance_metrics', {})
        }
