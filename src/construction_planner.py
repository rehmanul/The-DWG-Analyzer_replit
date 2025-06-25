"""
Construction Plan Generator with 2D/3D Visualization
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import streamlit as st

class ConstructionPlanner:
    def __init__(self):
        self.plan_data = {}
        self.box_placements = {}
        self.colors = {
            'wall': '#2C3E50',
            'door': '#E74C3C',
            'window': '#3498DB',
            'furniture': '#F39C12',
            'empty_space': '#ECF0F1',
            'box': '#E67E22'
        }
    
    def create_empty_plan(self, zones: List[Dict]) -> Dict:
        """Create empty construction plan from zones"""
        plan = {
            'zones': [],
            'walls': [],
            'doors': [],
            'windows': [],
            'dimensions': {}
        }
        
        for i, zone in enumerate(zones):
            points = zone.get('points', [])
            if len(points) < 3:
                continue
                
            # Create zone structure
            zone_plan = {
                'id': i,
                'name': zone.get('zone_type', f'Room {i+1}'),
                'points': points,
                'area': zone.get('area', 0),
                'walls': self._generate_walls_from_points(points),
                'doors': self._detect_doors(points),
                'windows': self._detect_windows(points)
            }
            plan['zones'].append(zone_plan)
        
        return plan
    
    def _generate_walls_from_points(self, points: List[Tuple]) -> List[Dict]:
        """Generate wall segments from room points"""
        walls = []
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            
            wall = {
                'start': start,
                'end': end,
                'length': np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2),
                'thickness': 0.2  # Default wall thickness
            }
            walls.append(wall)
        
        return walls
    
    def _detect_doors(self, points: List[Tuple]) -> List[Dict]:
        """Detect potential door locations"""
        doors = []
        # Simple door detection - place doors on longer walls
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            
            if length > 3.0:  # Only on walls longer than 3 units
                # Place door at 1/3 of wall length
                door_pos = (
                    start[0] + (end[0] - start[0]) * 0.33,
                    start[1] + (end[1] - start[1]) * 0.33
                )
                doors.append({
                    'position': door_pos,
                    'width': 0.9,
                    'wall_index': i
                })
        
        return doors[:1]  # Limit to 1 door per room
    
    def _detect_windows(self, points: List[Tuple]) -> List[Dict]:
        """Detect potential window locations"""
        windows = []
        # Place windows on exterior walls (simplified)
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            
            if length > 2.0:  # Only on walls longer than 2 units
                # Place window at 2/3 of wall length
                window_pos = (
                    start[0] + (end[0] - start[0]) * 0.66,
                    start[1] + (end[1] - start[1]) * 0.66
                )
                windows.append({
                    'position': window_pos,
                    'width': 1.2,
                    'wall_index': i
                })
        
        return windows[:2]  # Limit to 2 windows per room
    
    def create_2d_plan_visualization(self, plan: Dict, show_boxes: bool = False) -> go.Figure:
        """Create 2D construction plan visualization"""
        fig = go.Figure()
        
        # Draw zones (rooms)
        for zone in plan['zones']:
            points = zone['points']
            x_coords = [p[0] for p in points] + [points[0][0]]
            y_coords = [p[1] for p in points] + [points[0][1]]
            
            # Room outline
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color=self.colors['wall'], width=3),
                fill='toself',
                fillcolor='rgba(236, 240, 241, 0.3)',
                name=zone['name'],
                hovertemplate=f"<b>{zone['name']}</b><br>Area: {zone['area']:.1f} m²<extra></extra>"
            ))
            
            # Add room label
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            fig.add_annotation(
                x=centroid_x,
                y=centroid_y,
                text=zone['name'],
                showarrow=False,
                font=dict(size=12, color='black'),
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
            
            # Draw doors
            for door in zone.get('doors', []):
                fig.add_trace(go.Scatter(
                    x=[door['position'][0]],
                    y=[door['position'][1]],
                    mode='markers',
                    marker=dict(
                        symbol='square',
                        size=15,
                        color=self.colors['door']
                    ),
                    name='Door',
                    showlegend=False,
                    hovertemplate="<b>Door</b><br>Width: 0.9m<extra></extra>"
                ))
            
            # Draw windows
            for window in zone.get('windows', []):
                fig.add_trace(go.Scatter(
                    x=[window['position'][0]],
                    y=[window['position'][1]],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=12,
                        color=self.colors['window']
                    ),
                    name='Window',
                    showlegend=False,
                    hovertemplate="<b>Window</b><br>Width: 1.2m<extra></extra>"
                ))
        
        # Add boxes if requested
        if show_boxes and hasattr(self, 'box_placements'):
            self._add_boxes_to_2d_plan(fig)
        
        # Update layout
        fig.update_layout(
            title="2D Construction Plan",
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            showlegend=True,
            width=800,
            height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_3d_plan_visualization(self, plan: Dict, show_boxes: bool = False) -> go.Figure:
        """Create 3D construction plan visualization"""
        fig = go.Figure()
        
        # Draw 3D rooms
        for zone in plan['zones']:
            points = zone['points']
            
            # Create 3D room (floor and walls)
            self._add_3d_room(fig, points, zone['name'])
            
            # Add doors and windows in 3D
            for door in zone.get('doors', []):
                self._add_3d_door(fig, door)
            
            for window in zone.get('windows', []):
                self._add_3d_window(fig, window)
        
        # Add 3D boxes if requested
        if show_boxes and hasattr(self, 'box_placements'):
            self._add_boxes_to_3d_plan(fig)
        
        # Update 3D layout
        fig.update_layout(
            title="3D Construction Plan",
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)",
                zaxis_title="Z (meters)",
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _add_3d_room(self, fig: go.Figure, points: List[Tuple], name: str):
        """Add 3D room to figure"""
        # Floor
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [0] * len(points)
        
        fig.add_trace(go.Mesh3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            color='lightgray',
            opacity=0.3,
            name=f"{name} Floor"
        ))
        
        # Walls
        wall_height = 3.0
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            
            # Wall vertices
            wall_x = [start[0], end[0], end[0], start[0]]
            wall_y = [start[1], end[1], end[1], start[1]]
            wall_z = [0, 0, wall_height, wall_height]
            
            fig.add_trace(go.Mesh3d(
                x=wall_x,
                y=wall_y,
                z=wall_z,
                color=self.colors['wall'],
                opacity=0.7,
                name=f"{name} Wall",
                showlegend=False
            ))
    
    def _add_3d_door(self, fig: go.Figure, door: Dict):
        """Add 3D door to figure"""
        pos = door['position']
        width = door['width']
        
        # Simple door representation
        fig.add_trace(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[1.0],
            mode='markers',
            marker=dict(
                size=8,
                color=self.colors['door'],
                symbol='square'
            ),
            name='Door',
            showlegend=False
        ))
    
    def _add_3d_window(self, fig: go.Figure, window: Dict):
        """Add 3D window to figure"""
        pos = window['position']
        
        # Simple window representation
        fig.add_trace(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[1.5],
            mode='markers',
            marker=dict(
                size=6,
                color=self.colors['window'],
                symbol='diamond'
            ),
            name='Window',
            showlegend=False
        ))
    
    def _add_boxes_to_2d_plan(self, fig: go.Figure):
        """Add furniture boxes to 2D plan"""
        if not hasattr(self, 'analysis_results'):
            return
            
        placements = self.analysis_results.get('placements', {})
        for zone_name, boxes in placements.items():
            for box in boxes:
                pos = box['position']
                size = box['size']
                
                # Draw box as rectangle
                box_x = [pos[0] - size[0]/2, pos[0] + size[0]/2, 
                        pos[0] + size[0]/2, pos[0] - size[0]/2, pos[0] - size[0]/2]
                box_y = [pos[1] - size[1]/2, pos[1] - size[1]/2,
                        pos[1] + size[1]/2, pos[1] + size[1]/2, pos[1] - size[1]/2]
                
                fig.add_trace(go.Scatter(
                    x=box_x,
                    y=box_y,
                    mode='lines',
                    line=dict(color=self.colors['box'], width=2),
                    fill='toself',
                    fillcolor='rgba(230, 126, 34, 0.5)',
                    name='Furniture',
                    showlegend=False,
                    hovertemplate=f"<b>Furniture Box</b><br>Size: {size[0]:.1f}x{size[1]:.1f}m<extra></extra>"
                ))
    
    def _add_boxes_to_3d_plan(self, fig: go.Figure):
        """Add furniture boxes to 3D plan"""
        if not hasattr(self, 'analysis_results'):
            return
            
        placements = self.analysis_results.get('placements', {})
        for zone_name, boxes in placements.items():
            for box in boxes:
                pos = box['position']
                size = box['size']
                
                # Create 3D box
                box_height = 0.8  # Standard furniture height
                
                # Box vertices
                x_coords = [pos[0] - size[0]/2, pos[0] + size[0]/2] * 4
                y_coords = [pos[1] - size[1]/2] * 2 + [pos[1] + size[1]/2] * 2
                y_coords = y_coords * 2
                z_coords = [0] * 4 + [box_height] * 4
                
                fig.add_trace(go.Mesh3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    color=self.colors['box'],
                    opacity=0.7,
                    name='Furniture',
                    showlegend=False
                ))
    
    def generate_construction_report(self, plan: Dict) -> Dict:
        """Generate construction report with measurements and specifications"""
        report = {
            'total_rooms': len(plan['zones']),
            'total_area': sum(zone['area'] for zone in plan['zones']),
            'room_details': [],
            'materials_needed': {},
            'cost_estimate': {}
        }
        
        for zone in plan['zones']:
            room_detail = {
                'name': zone['name'],
                'area': zone['area'],
                'perimeter': self._calculate_perimeter(zone['points']),
                'doors': len(zone.get('doors', [])),
                'windows': len(zone.get('windows', [])),
                'wall_length': sum(wall['length'] for wall in zone.get('walls', []))
            }
            report['room_details'].append(room_detail)
        
        # Calculate materials
        total_wall_length = sum(
            sum(wall['length'] for wall in zone.get('walls', []))
            for zone in plan['zones']
        )
        
        report['materials_needed'] = {
            'concrete_blocks': int(total_wall_length * 20),  # blocks per meter
            'cement_bags': int(total_wall_length * 2),
            'steel_bars': int(total_wall_length * 5),
            'doors': sum(len(zone.get('doors', [])) for zone in plan['zones']),
            'windows': sum(len(zone.get('windows', [])) for zone in plan['zones'])
        }
        
        return report
    
    def _calculate_perimeter(self, points: List[Tuple]) -> float:
        """Calculate perimeter of a polygon"""
        perimeter = 0
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            perimeter += np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        return perimeter
    
    def set_analysis_results(self, results: Dict):
        """Set analysis results for box placement"""
        self.analysis_results = results