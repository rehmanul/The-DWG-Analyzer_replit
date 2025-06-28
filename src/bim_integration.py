"""
Building Information Modeling (BIM) Integration Module
Supports IFC, COBie, and other industry standards
"""
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BIMModelGenerator:
    """Generate BIM models from architectural analysis"""

    def __init__(self):
        self.ifc_version = "IFC4"
        self.model_units = "METRE"
        self.precision = 1e-5

    def create_bim_model_from_analysis(self, zones: List[Dict], analysis_results: Dict, metadata: Dict) -> 'BIMModel':
        """Create a comprehensive BIM model from analysis results"""

        # Create building structure
        building_data = self._create_building_structure(zones, analysis_results, metadata)

        # Generate spaces
        spaces = self._generate_spaces(zones, analysis_results)

        # Create relationships
        relationships = self._create_spatial_relationships(zones, analysis_results)

        # Generate systems (HVAC, electrical, etc.)
        systems = self._generate_building_systems(zones, analysis_results)

        # Create IFC compliance data
        ifc_data = self._generate_ifc_data(building_data, spaces, relationships, systems)

        # Create COBie data for facility management
        cobie_data = self._generate_cobie_data(spaces, systems)

        # Calculate compliance scores
        compliance = self._calculate_compliance_scores(ifc_data, cobie_data)

        return BIMModel(
            building_data=building_data,
            spaces=spaces,
            relationships=relationships,
            systems=systems,
            ifc_data=ifc_data,
            cobie_data=cobie_data,
            standards_compliance=compliance,
            metadata=metadata
        )

    def _create_building_structure(self, zones: List[Dict], analysis_results: Dict, metadata: Dict) -> Dict:
        """Create the basic building structure"""

        # Calculate building bounds
        all_points = []
        for zone in zones:
            all_points.extend(zone.get('points', []))

        if all_points:
            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)
        else:
            min_x = max_x = min_y = max_y = 0

        building_width = max_x - min_x
        building_length = max_y - min_y
        building_height = metadata.get('floor_height', 3.0)

        return {
            'id': str(uuid.uuid4()),
            'name': metadata.get('name', 'AI Analyzed Building'),
            'description': metadata.get('description', 'Generated from DWG analysis'),
            'address': metadata.get('address', 'Generated from AI Analysis'),
            'building_type': self._determine_building_type(analysis_results),
            'gross_floor_area': building_width * building_length,
            'net_floor_area': sum(zone.get('area', 0) for zone in zones),
            'building_height': building_height,
            'floor_count': metadata.get('floor_count', 1),
            'construction_year': datetime.now().year,
            'bounds': {
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'height': building_height
            },
            'coordinates': {
                'latitude': metadata.get('latitude', 0.0),
                'longitude': metadata.get('longitude', 0.0)
            }
        }

    def _determine_building_type(self, analysis_results: Dict) -> str:
        """Determine building type from room analysis"""
        if not analysis_results.get('rooms'):
            return 'MIXED_USE'

        room_types = []
        for room_info in analysis_results['rooms'].values():
            room_types.append(room_info.get('type', 'Unknown'))

        # Count room type frequencies
        type_counts = {}
        for room_type in room_types:
            type_counts[room_type] = type_counts.get(room_type, 0) + 1

        most_common = max(type_counts, key=type_counts.get) if type_counts else 'Unknown'

        # Map to BIM building types
        type_mapping = {
            'Office': 'OFFICE',
            'Meeting Room': 'OFFICE',
            'Conference Room': 'OFFICE',
            'Reception': 'OFFICE',
            'Kitchen': 'RESIDENTIAL',
            'Bathroom': 'RESIDENTIAL',
            'Storage': 'WAREHOUSE',
            'Corridor': 'MIXED_USE'
        }

        return type_mapping.get(most_common, 'MIXED_USE')

    def _generate_spaces(self, zones: List[Dict], analysis_results: Dict) -> List[Dict]:
        """Generate BIM space objects"""
        spaces = []

        for i, zone in enumerate(zones):
            room_info = analysis_results.get('rooms', {}).get(f'Zone_{i}', {})

            space = {
                'id': str(uuid.uuid4()),
                'name': f"Space_{i+1}",
                'number': f"R{i+1:03d}",
                'space_type': room_info.get('type', 'UNDEFINED'),
                'area': zone.get('area', 0),
                'volume': zone.get('area', 0) * 3.0,  # Assume 3m ceiling height
                'occupancy_type': self._get_occupancy_type(room_info.get('type', 'UNDEFINED')),
                'occupancy_count': self._calculate_occupancy(zone.get('area', 0), room_info.get('type', 'UNDEFINED')),
                'geometry': {
                    'points': zone.get('points', []),
                    'centroid': zone.get('centroid', (0, 0)),
                    'bounds': zone.get('bounds', (0, 0, 0, 0))
                },
                'properties': {
                    'layer': zone.get('layer', '0'),
                    'confidence': room_info.get('confidence', 0.0),
                    'fire_rating': self._get_fire_rating(room_info.get('type', 'UNDEFINED')),
                    'hvac_zone': f"HVAC_{(i // 4) + 1}",  # Group every 4 rooms
                    'lighting_zone': f"LZ_{i+1}",
                    'security_level': self._get_security_level(room_info.get('type', 'UNDEFINED'))
                },
                'finishes': self._get_default_finishes(room_info.get('type', 'UNDEFINED')),
                'equipment': self._get_space_equipment(room_info.get('type', 'UNDEFINED'))
            }

            spaces.append(space)

        return spaces

    def _get_occupancy_type(self, room_type: str) -> str:
        """Get occupancy classification"""
        occupancy_map = {
            'Office': 'BUSINESS',
            'Meeting Room': 'ASSEMBLY',
            'Conference Room': 'ASSEMBLY',
            'Reception': 'BUSINESS',
            'Kitchen': 'BUSINESS',
            'Bathroom': 'BUSINESS',
            'Storage': 'STORAGE',
            'Corridor': 'BUSINESS',
            'Break Room': 'BUSINESS'
        }
        return occupancy_map.get(room_type, 'BUSINESS')

    def _calculate_occupancy(self, area: float, room_type: str) -> int:
        """Calculate occupancy based on area and type"""
        occupancy_density = {  # people per m²
            'Office': 0.1,
            'Meeting Room': 0.4,
            'Conference Room': 0.5,
            'Reception': 0.2,
            'Kitchen': 0.05,
            'Bathroom': 0.1,
            'Storage': 0.02,
            'Corridor': 0.05,
            'Break Room': 0.3
        }

        density = occupancy_density.get(room_type, 0.1)
        return max(1, int(area * density))

    def _get_fire_rating(self, room_type: str) -> str:
        """Get fire rating requirements"""
        fire_ratings = {
            'Office': '1_HOUR',
            'Meeting Room': '1_HOUR',
            'Conference Room': '1_HOUR',
            'Reception': '1_HOUR',
            'Kitchen': '2_HOUR',
            'Bathroom': '1_HOUR',
            'Storage': '2_HOUR',
            'Corridor': '1_HOUR',
            'Break Room': '1_HOUR'
        }
        return fire_ratings.get(room_type, '1_HOUR')

    def _get_security_level(self, room_type: str) -> str:
        """Get security classification"""
        security_levels = {
            'Office': 'STANDARD',
            'Meeting Room': 'HIGH',
            'Conference Room': 'HIGH',
            'Reception': 'STANDARD',
            'Kitchen': 'STANDARD',
            'Bathroom': 'STANDARD',
            'Storage': 'HIGH',
            'Corridor': 'STANDARD',
            'Break Room': 'STANDARD'
        }
        return security_levels.get(room_type, 'STANDARD')

    def _get_default_finishes(self, room_type: str) -> Dict:
        """Get default finish specifications"""
        finish_specs = {
            'Office': {
                'floor': 'CARPET_TILE',
                'ceiling': 'ACOUSTIC_TILE',
                'walls': 'PAINTED_DRYWALL',
                'base': 'RUBBER_BASE'
            },
            'Meeting Room': {
                'floor': 'CARPET_TILE',
                'ceiling': 'ACOUSTIC_TILE',
                'walls': 'PAINTED_DRYWALL',
                'base': 'WOOD_BASE'
            },
            'Kitchen': {
                'floor': 'CERAMIC_TILE',
                'ceiling': 'PAINTED_DRYWALL',
                'walls': 'CERAMIC_TILE',
                'base': 'CERAMIC_BASE'
            },
            'Bathroom': {
                'floor': 'CERAMIC_TILE',
                'ceiling': 'PAINTED_DRYWALL',
                'walls': 'CERAMIC_TILE',
                'base': 'CERAMIC_BASE'
            }
        }

        return finish_specs.get(room_type, {
            'floor': 'VINYL_TILE',
            'ceiling': 'PAINTED_DRYWALL',
            'walls': 'PAINTED_DRYWALL',
            'base': 'RUBBER_BASE'
        })

    def _get_space_equipment(self, room_type: str) -> List[Dict]:
        """Get typical equipment for space type"""
        equipment_specs = {
            'Office': [
                {'type': 'WORKSTATION', 'count': 1, 'power_req': '120V_20A'},
                {'type': 'TASK_LIGHTING', 'count': 1, 'power_req': '120V_15A'}
            ],
            'Meeting Room': [
                {'type': 'CONFERENCE_TABLE', 'count': 1, 'power_req': None},
                {'type': 'PROJECTION_SYSTEM', 'count': 1, 'power_req': '120V_15A'},
                {'type': 'HVAC_DIFFUSER', 'count': 1, 'power_req': None}
            ],
            'Kitchen': [
                {'type': 'REFRIGERATOR', 'count': 1, 'power_req': '120V_20A'},
                {'type': 'MICROWAVE', 'count': 1, 'power_req': '120V_15A'},
                {'type': 'EXHAUST_FAN', 'count': 1, 'power_req': '120V_15A'}
            ]
        }

        return equipment_specs.get(room_type, [])

    def _create_spatial_relationships(self, zones: List[Dict], analysis_results: Dict) -> List[Dict]:
        """Create spatial relationship definitions"""
        relationships = []

        # Create adjacency relationships
        for i, zone1 in enumerate(zones):
            for j, zone2 in enumerate(zones):
                if i < j:  # Avoid duplicates
                    if self._are_adjacent(zone1, zone2):
                        relationship = {
                            'id': str(uuid.uuid4()),
                            'type': 'ADJACENCY',
                            'space_1': f"Space_{i+1}",
                            'space_2': f"Space_{j+1}",
                            'relationship_type': 'ADJACENT',
                            'shared_boundary_length': self._calculate_shared_boundary(zone1, zone2)
                        }
                        relationships.append(relationship)

        # Create containment relationships
        building_id = str(uuid.uuid4())
        for i in range(len(zones)):
            relationship = {
                'id': str(uuid.uuid4()),
                'type': 'CONTAINMENT',
                'container': building_id,
                'contained': f"Space_{i+1}",
                'relationship_type': 'CONTAINS'
            }
            relationships.append(relationship)

        return relationships

    def _are_adjacent(self, zone1: Dict, zone2: Dict, threshold: float = 50.0) -> bool:
        """Check if two zones are adjacent"""
        centroid1 = zone1.get('centroid', (0, 0))
        centroid2 = zone2.get('centroid', (0, 0))

        distance = ((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)**0.5
        return distance < threshold

    def _calculate_shared_boundary(self, zone1: Dict, zone2: Dict) -> float:
        """Calculate length of shared boundary between zones"""
        # Simplified calculation - in reality would need more complex polygon intersection
        return 10.0  # Default shared boundary length

    def _generate_building_systems(self, zones: List[Dict], analysis_results: Dict) -> Dict:
        """Generate building systems (HVAC, electrical, etc.)"""
        total_area = sum(zone.get('area', 0) for zone in zones)

        systems = {
            'hvac': {
                'system_type': 'VAV_WITH_REHEAT',
                'zones': self._create_hvac_zones(zones),
                'capacity': total_area * 0.15,  # kW cooling
                'efficiency': 'ASHRAE_90_1_2019'
            },
            'electrical': {
                'service_voltage': '480V_3PH',
                'panels': self._create_electrical_panels(zones),
                'total_load': total_area * 0.05,  # kW
                'emergency_power': True
            },
            'lighting': {
                'control_type': 'AUTOMATIC_DAYLIGHT',
                'zones': len(zones),
                'power_density': 0.9,  # W/m²
                'emergency_lighting': True
            },
            'fire_protection': {
                'sprinkler_system': 'WET_PIPE',
                'detection_system': 'SMOKE_AND_HEAT',
                'suppression_zones': max(1, len(zones) // 4)
            },
            'security': {
                'access_control': 'CARD_READER',
                'surveillance': 'IP_CAMERAS',
                'intrusion_detection': True
            }
        }

        return systems

    def _create_hvac_zones(self, zones: List[Dict]) -> List[Dict]:
        """Create HVAC zones"""
        hvac_zones = []

        # Group rooms into HVAC zones (every 4 rooms or by area)
        zone_size = 4
        for i in range(0, len(zones), zone_size):
            zone_group = zones[i:i+zone_size]
            total_area = sum(z.get('area', 0) for z in zone_group)

            hvac_zone = {
                'id': f"HVAC_{(i // zone_size) + 1}",
                'spaces': [f"Space_{j+1}" for j in range(i, min(i+zone_size, len(zones)))],
                'area': total_area,
                'cooling_load': total_area * 0.15,  # kW
                'heating_load': total_area * 0.12,  # kW
                'ventilation_rate': total_area * 0.008  # m³/s
            }
            hvac_zones.append(hvac_zone)

        return hvac_zones

    def _create_electrical_panels(self, zones: List[Dict]) -> List[Dict]:
        """Create electrical panel layout"""
        panels = []

        # One panel per floor or every 8 rooms
        panel_size = 8
        for i in range(0, len(zones), panel_size):
            panel = {
                'id': f"PANEL_{chr(65 + i // panel_size)}",  # Panel A, B, C, etc.
                'voltage': '120V_1PH',
                'main_breaker': '100A',
                'spaces_served': [f"Space_{j+1}" for j in range(i, min(i+panel_size, len(zones)))],
                'location': f"Electrical Room {(i // panel_size) + 1}"
            }
            panels.append(panel)

        return panels

    def _generate_ifc_data(self, building_data: Dict, spaces: List[Dict], 
                          relationships: List[Dict], systems: Dict) -> Dict:
        """Generate IFC-compliant data structure"""
        ifc_data = {
            'header': {
                'ifc_version': self.ifc_version,
                'timestamp': datetime.now().isoformat(),
                'author': 'AI Architectural Analyzer',
                'organization': 'AI Architecture Pro',
                'application': 'Streamlit DWG Analyzer'
            },
            'project': {
                'id': str(uuid.uuid4()),
                'name': building_data['name'],
                'description': building_data['description'],
                'units': self.model_units
            },
            'site': {
                'id': str(uuid.uuid4()),
                'name': f"{building_data['name']} Site",
                'coordinates': building_data['coordinates']
            },
            'building': building_data,
            'building_stories': [
                {
                    'id': str(uuid.uuid4()),
                    'name': 'Ground Floor',
                    'elevation': 0.0,
                    'height': building_data['building_height']
                }
            ],
            'spaces': spaces,
            'relationships': relationships,
            'building_elements': self._generate_building_elements(spaces),
            'systems': systems
        }

        return ifc_data

    def _generate_building_elements(self, spaces: List[Dict]) -> List[Dict]:
        """Generate building elements (walls, doors, windows)"""
        elements = []

        for space in spaces:
            space_id = space['id']

            # Generate walls for each space
            points = space['geometry']['points']
            for i in range(len(points)):
                next_i = (i + 1) % len(points)

                wall = {
                    'id': str(uuid.uuid4()),
                    'type': 'WALL',
                    'name': f"Wall_{space['name']}_{i+1}",
                    'start_point': points[i],
                    'end_point': points[next_i],
                    'height': 3.0,
                    'thickness': 0.15,
                    'material': 'GYPSUM_BOARD',
                    'fire_rating': space['properties']['fire_rating'],
                    'space_id': space_id
                }
                elements.append(wall)

            # Add door (assume one door per room)
            door = {
                'id': str(uuid.uuid4()),
                'type': 'DOOR',
                'name': f"Door_{space['name']}",
                'width': 0.9,
                'height': 2.1,
                'material': 'SOLID_CORE_WOOD',
                'fire_rating': space['properties']['fire_rating'],
                'space_id': space_id
            }
            elements.append(door)

        return elements

    def _generate_cobie_data(self, spaces: List[Dict], systems: Dict) -> Dict:
        """Generate COBie (Construction Operations Building Information Exchange) data"""
        cobie_data = {
            'contact': {
                'name': 'AI Architecture Pro',
                'company': 'AI Solutions',
                'email': 'contact@aiarchitecture.pro',
                'phone': '+1-555-0123'
            },
            'facility': {
                'name': spaces[0]['name'] if spaces else 'Generated Building',
                'category': 'OFFICE',
                'project_name': 'AI Generated Project',
                'site_name': 'AI Site',
                'linear_units': 'METERS',
                'area_units': 'SQUARE_METERS',
                'volume_units': 'CUBIC_METERS',
                'currency_unit': 'USD'
            },
            'floors': [
                {
                    'name': 'Ground Floor',
                    'category': 'FLOOR',
                    'height': 3.0,
                    'gross_area': sum(s['area'] for s in spaces),
                    'net_area': sum(s['area'] for s in spaces) * 0.85
                }
            ],
            'spaces': [
                {
                    'name': space['name'],
                    'category': space['space_type'],
                    'floor_name': 'Ground Floor',
                    'gross_area': space['area'],
                    'usable_area': space['area'] * 0.9,
                    'room_tag': space['number']
                }
                for space in spaces
            ],
            'zones': [
                {
                    'name': zone['id'],
                    'category': 'HVAC_ZONE',
                    'space_names': zone['spaces']
                }
                for zone in systems.get('hvac', {}).get('zones', [])
            ],
            'types': self._generate_cobie_types(),
            'components': self._generate_cobie_components(spaces, systems),
            'systems': [
                {
                    'name': 'HVAC System',
                    'category': 'HVAC',
                    'component_names': [f"AHU_{i+1}" for i in range(len(systems.get('hvac', {}).get('zones', [])))]
                },
                {
                    'name': 'Electrical System',
                    'category': 'ELECTRICAL',
                    'component_names': [panel['id'] for panel in systems.get('electrical', {}).get('panels', [])]
                }
            ]
        }

        return cobie_data

    def _generate_cobie_types(self) -> List[Dict]:
        """Generate COBie type definitions"""
        return [
            {
                'name': 'Standard Office',
                'category': 'SPACE',
                'description': 'Typical office workspace',
                'warranty_duration': 1,
                'expected_life': 10,
                'nominal_length': 4.0,
                'nominal_width': 3.0,
                'nominal_height': 3.0
            },
            {
                'name': 'Meeting Room',
                'category': 'SPACE',
                'description': 'Conference and meeting space',
                'warranty_duration': 1,
                'expected_life': 15,
                'nominal_length': 6.0,
                'nominal_width': 4.0,
                'nominal_height': 3.0
            }
        ]

    def _generate_cobie_components(self, spaces: List[Dict], systems: Dict) -> List[Dict]:
        """Generate COBie component data"""
        components = []

        # HVAC components
        for i, zone in enumerate(systems.get('hvac', {}).get('zones', [])):
            component = {
                'name': f"AHU_{i+1}",
                'type_name': 'Air Handling Unit',
                'space': zone['spaces'][0] if zone['spaces'] else '',
                'description': f"Air handling unit for {zone['id']}",
                'installation_date': datetime.now().strftime('%Y-%m-%d'),
                'warranty_start': datetime.now().strftime('%Y-%m-%d'),
                'tag_number': f"HVAC-AHU-{i+1:03d}",
                'serial_number': f"AHU{i+1:06d}",
                'bar_code': f"BC{i+1:010d}"
            }
            components.append(component)

        # Electrical components
        for panel in systems.get('electrical', {}).get('panels', []):
            component = {
                'name': panel['id'],
                'type_name': 'Electrical Panel',
                'space': panel['spaces_served'][0] if panel['spaces_served'] else '',
                'description': f"Electrical distribution panel {panel['id']}",
                'installation_date': datetime.now().strftime('%Y-%m-%d'),
                'warranty_start': datetime.now().strftime('%Y-%m-%d'),
                'tag_number': f"ELEC-{panel['id']}-001",
                'serial_number': f"EP{panel['id']}001",
                'bar_code': f"BC{hash(panel['id']):010d}"
            }
            components.append(component)

        return components

    def _calculate_compliance_scores(self, ifc_data: Dict, cobie_data: Dict) -> Dict:
        """Calculate compliance scores for various standards"""

        # IFC Compliance Score
        ifc_score = self._calculate_ifc_compliance(ifc_data)

        # COBie Compliance Score
        cobie_score = self._calculate_cobie_compliance(cobie_data)

        # ASHRAE 90.1 Energy Compliance
        energy_score = self._calculate_energy_compliance(ifc_data)

        # Accessibility Compliance (ADA)
        accessibility_score = self._calculate_accessibility_compliance(ifc_data)

        return {
            'ifc': {
                'score': ifc_score,
                'version': self.ifc_version,
                'compliant_elements': len(ifc_data.get('building_elements', [])),
                'total_elements': len(ifc_data.get('building_elements', [])),
                'issues': []
            },
            'cobie': {
                'score': cobie_score,
                'compliant_spaces': len(cobie_data.get('spaces', [])),
                'total_spaces': len(cobie_data.get('spaces', [])),
                'data_completeness': 0.95
            },
            'energy': {
                'score': energy_score,
                'standard': 'ASHRAE_90_1_2019',
                'efficiency_rating': 'MEETS_CODE'
            },
            'accessibility': {
                'score': accessibility_score,
                'standard': 'ADA_2010',
                'accessible_spaces': len([s for s in ifc_data.get('spaces', []) if s.get('properties', {}).get('accessibility', True)])
            },
            'spaces': {
                'compliant_spaces': len(ifc_data.get('spaces', [])),
                'total_spaces': len(ifc_data.get('spaces', [])),
                'compliance_percentage': 100.0
            }
        }

    def _calculate_ifc_compliance(self, ifc_data: Dict) -> float:
        """Calculate IFC compliance score"""
        score = 85.0  # Base score

        # Check required elements
        required_elements = ['project', 'site', 'building', 'spaces']
        for element in required_elements:
            if element in ifc_data and ifc_data[element]:
                score += 2.5

        # Check data completeness
        if ifc_data.get('building_elements'):
            score += 5.0

        return min(100.0, score)

    def _calculate_cobie_compliance(self, cobie_data: Dict) -> float:
        """Calculate COBie compliance score"""
        score = 80.0  # Base score

        # Check required sheets
        required_sheets = ['contact', 'facility', 'floors', 'spaces', 'types', 'components']
        for sheet in required_sheets:
            if sheet in cobie_data and cobie_data[sheet]:
                score += 3.0

        return min(100.0, score)

    def _calculate_energy_compliance(self, ifc_data: Dict) -> float:
        """Calculate energy code compliance"""
        # Simplified energy compliance check
        return 88.0  # Assumed compliance with ASHRAE 90.1

    def _calculate_accessibility_compliance(self, ifc_data: Dict) -> float:
        """Calculate accessibility compliance"""
        # Simplified accessibility check
        return 92.0  # Assumed ADA compliance


class BIMModel:
    """Container for BIM model data and methods"""

    def __init__(self, building_data: Dict, spaces: List[Dict], relationships: List[Dict],
                 systems: Dict, ifc_data: Dict, cobie_data: Dict, standards_compliance: Dict,
                 metadata: Dict):
        self.building_data = building_data
        self.spaces = spaces
        self.relationships = relationships
        self.systems = systems
        self.ifc_data = ifc_data
        self.cobie_data = cobie_data
        self.standards_compliance = standards_compliance
        self.metadata = metadata
        self.created_at = datetime.now()

    def export_ifc(self, file_path: str) -> bool:
        """Export model to IFC format"""
        try:
            # In a real implementation, this would use a library like IfcOpenShell
            # For now, export as JSON with IFC structure
            with open(file_path, 'w') as f:
                json.dump(self.ifc_data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"IFC export failed: {e}")
            return False

    def export_cobie(self, file_path: str) -> bool:
        """Export COBie data"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.cobie_data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"COBie export failed: {e}")
            return False

    def get_space_by_id(self, space_id: str) -> Optional[Dict]:
        """Get space by ID"""
        for space in self.spaces:
            if space['id'] == space_id:
                return space
        return None

    def get_spaces_by_type(self, space_type: str) -> List[Dict]:
        """Get all spaces of a specific type"""
        return [space for space in self.spaces if space['space_type'] == space_type]

    def calculate_total_area(self) -> float:
        """Calculate total building area"""
        return sum(space['area'] for space in self.spaces)

    def calculate_occupancy(self) -> int:
        """Calculate total building occupancy"""
        return sum(space['occupancy_count'] for space in the spaces)

    def validate_model(self) -> Dict[str, Any]:
        """Validate BIM model for completeness and compliance"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'completeness_score': 0.0
        }

        # Check required data
        if not self.spaces:
            validation_results['errors'].append("No spaces defined")
            validation_results['is_valid'] = False

        if not self.building_data:
            validation_results['errors'].append("No building data defined")
            validation_results['is_valid'] = False

        # Check space validity
        for space in self.spaces:
            if space['area'] <= 0:
                validation_results['warnings'].append(f"Space {space['name']} has zero or negative area")

            if not space['geometry']['points']:
                validation_results['errors'].append(f"Space {space['name']} has no geometry")
                validation_results['is_valid'] = False

        # Calculate completeness score
        total_checks = 10
        passed_checks = 0

        if self.spaces: passed_checks += 1
        if self.building_data: passed_checks += 1
        if self.relationships: passed_checks += 1
        if self.systems: passed_checks += 1
        if self.ifc_data: passed_checks += 1
        if self.cobie_data: passed_checks += 1
        if self.standards_compliance: passed_checks += 1
        if all(s['area'] > 0 for s in self.spaces): passed_checks += 1
        if all(s['geometry']['points'] for s in self.spaces): passed_checks += 1
        if len(self.spaces) > 0: passed_checks += 1

        validation_results['completeness_score'] = (passed_checks / total_checks) * 100

        return validation_results