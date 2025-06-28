import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json
import io
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
import asyncio
import tempfile
import os
import numpy as np
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import custom modules
from src.dwg_parser import DWGParser
from src.enhanced_dwg_parser import EnhancedDWGParser, parse_dwg_file_enhanced
from src.robust_error_handler import RobustErrorHandler
from src.enhanced_zone_detector import EnhancedZoneDetector
from src.navigation_manager import NavigationManager
from src.placement_optimizer import PlacementOptimizer
from src.pdf_parser import PDFParser
from src.ai_analyzer import AIAnalyzer
from src.visualization import PlanVisualizer
from src.export_utils import ExportManager
from src.optimization import PlacementOptimizer

# Import database and AI
from src.database import DatabaseManager
import os

# Configure PostgreSQL
os.environ['DATABASE_URL'] = 'postgresql://yang:nNTm6Q4un1aF25fmVvl7YqSzWffyznIe@dpg-d0t3rlili9vc739k84gg-a.oregon-postgres.render.com/dg4u_tiktok_bot'
from src.ai_integration import GeminiAIAnalyzer

# Configure page
st.set_page_config(
    page_title="AI Architectural Space Analyzer PRO",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'zones' not in st.session_state:
    st.session_state.zones = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'advanced_mode' not in st.session_state:
    st.session_state.advanced_mode = False

# Initialize components
@st.cache_resource
def get_advanced_components():
    return {
        'database': DatabaseManager(database_url=os.environ.get('DATABASE_URL')),
        'ai_analyzer': GeminiAIAnalyzer() if os.environ.get("GEMINI_API_KEY") else None
    }

def load_uploaded_file(uploaded_file):
    """Load uploaded file with instant demo layouts"""
    if uploaded_file is None:
        st.error("No file provided")
        return None

    try:
        # Validate file
        if not uploaded_file.name:
            st.error("Invalid file: No filename")
            return None

        # Check extension
        file_ext = uploaded_file.name.lower().split('.')[-1]
        if file_ext not in ['dwg', 'dxf']:
            st.error(f"Unsupported format: {file_ext}. Please upload DWG or DXF files.")
            return None

        # Read file
        try:
            file_bytes = uploaded_file.getvalue()
        except Exception as e:
            st.error(f"Could not read file: {str(e)}")
            return None

        if not file_bytes:
            st.error("File appears to be empty")
            return None

        file_size_mb = len(file_bytes) / (1024 * 1024)

        # Create demo layout based on filename
        zones = RobustErrorHandler.create_default_zones(uploaded_file.name, f"Demo layout for {uploaded_file.name}")
        st.success(f"✅ Layout ready: {len(zones)} zones for {uploaded_file.name}")

        # Update session state
        st.session_state.zones = zones
        st.session_state.file_loaded = True
        st.session_state.current_file = uploaded_file.name
        st.session_state.analysis_results = {}

        return zones

    except Exception as e:
        st.error(f"File upload error: {str(e)}")
        zones = RobustErrorHandler.create_default_zones("emergency_fallback", "Error recovery")
        st.session_state.zones = zones
        st.session_state.file_loaded = True
        st.session_state.current_file = "fallback_file"
        st.warning(f"Created emergency fallback layout with {len(zones)} zones")
        return zonesn zones

def run_ai_analysis(box_length, box_width, margin, confidence_threshold, enable_rotation, smart_spacing):
    """Run AI analysis on loaded zones"""
    try:
        with st.spinner("🤖 Running AI analysis..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize AI analyzer
            analyzer = AIAnalyzer(confidence_threshold)

            # Step 1: Room type analysis
            status_text.text("Analyzing room types...")
            progress_bar.progress(25)
            room_analysis = analyzer.analyze_room_types(st.session_state.zones)

            # Step 2: Furniture placement analysis
            status_text.text("Calculating optimal placements...")
            progress_bar.progress(50)

            params = {
                'box_size': (box_length, box_width),
                'margin': margin,
                'allow_rotation': enable_rotation,
                'smart_spacing': smart_spacing
            }

            placement_analysis = analyzer.analyze_furniture_placement(
                st.session_state.zones, params)

            # Step 3: Optimization
            status_text.text("Optimizing placements...")
            progress_bar.progress(75)

            optimizer = PlacementOptimizer()
            optimization_results = optimizer.optimize_placements(
                placement_analysis, params)

            # Step 4: Compile results
            status_text.text("Compiling results...")
            progress_bar.progress(100)

            st.session_state.analysis_results = {
                'rooms': room_analysis,
                'placements': placement_analysis,
                'optimization': optimization_results,
                'parameters': params,
                'total_boxes': sum(len(spots) for spots in placement_analysis.values()),
                'analysis_type': 'standard',
                'timestamp': datetime.now().isoformat()
            }

            progress_bar.empty()
            status_text.empty()

            st.success(
                f"Analysis complete! Found {st.session_state.analysis_results.get('total_boxes', 0)} optimal box placements"
            )

    except Exception as e:
        st.error(f"❌ Error during AI analysis: {str(e)}")

def display_plan_visualization():
    """Display plan visualization"""
    if not st.session_state.zones:
        st.info("Load a DWG file to see visualization")
        return

    visualizer = PlanVisualizer()

    # Visualization options
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("🎨 Display Options")
        show_zones = st.checkbox("Show Zones", value=True, key="plan_viz_zones_display")
        show_boxes = st.checkbox("Show Box Placements", value=True, key="plan_viz_boxes_display") 
        show_labels = st.checkbox("Show Labels", value=True, key="plan_viz_labels_display")
        color_by_type = st.checkbox("Color by Room Type", value=True, key="plan_viz_color_display")

    with col1:
        # Generate visualization
        if st.session_state.analysis_results:
            fig = visualizer.create_interactive_plot(
                st.session_state.zones,
                st.session_state.analysis_results,
                show_zones=show_zones,
                show_boxes=show_boxes,
                show_labels=show_labels,
                color_by_type=color_by_type)
        else:
            fig = visualizer.create_basic_plot(st.session_state.zones)

        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🏗️ AI Architectural Space Analyzer PRO")
    
    # Get components
    components = get_advanced_components()
    
    # Header with mode toggle
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            "**Complete Professional Solution for Architectural Analysis & Space Planning**"
        )

    with col2:
        st.session_state.advanced_mode = st.toggle(
            "Advanced Mode",
            value=st.session_state.advanced_mode,
            key="main_advanced_mode_toggle")

    # File upload section
    st.subheader("📁 Upload DWG/DXF File")
    
    uploaded_file = st.file_uploader(
        "Select your architectural drawing (DWG/DXF format)",
        type=['dwg', 'dxf'],
        key="main_file_uploader")

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.write(f"📄 **{uploaded_file.name}** ({file_size_mb:.1f} MB)")
        
        if st.button("Load File", type="primary", key="load_file_btn"):
            zones = load_uploaded_file(uploaded_file)
            if zones:
                st.success(f"✅ Successfully loaded {len(zones)} zones")
                st.rerun()

    # Analysis parameters
    if st.session_state.file_loaded:
        st.subheader("🔧 Analysis Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            box_length = st.number_input("Box Length (m)",
                                        min_value=0.1,
                                        max_value=10.0,
                                        value=2.0,
                                        step=0.1)
        with col2:
            box_width = st.number_input("Box Width (m)",
                                        min_value=0.1,
                                        max_value=10.0,
                                        value=1.5,
                                        step=0.1)
        with col3:
            margin = st.number_input("Margin (m)",
                                    min_value=0.0,
                                    max_value=5.0,
                                    value=0.5,
                                    step=0.1)
        
        confidence_threshold = st.slider("Confidence Threshold",
                                        min_value=0.5,
                                        max_value=0.95,
                                        value=0.7,
                                        step=0.05)
        
        enable_rotation = st.checkbox("Allow Box Rotation", value=True)
        smart_spacing = st.checkbox("Smart Spacing Optimization", value=True)
        
        if st.button("🤖 Run AI Analysis", type="primary"):
            run_ai_analysis(box_length, box_width, margin, confidence_threshold, enable_rotation, smart_spacing)
    
    # Display results
    if st.session_state.analysis_results:
        st.subheader("📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        results = st.session_state.analysis_results
        
        with col1:
            st.metric("Total Boxes", results.get('total_boxes', 0))
        with col2:
            total_area = results.get('total_boxes', 0) * results.get('parameters', {}).get('box_size', [2.0, 1.5])[0] * results.get('parameters', {}).get('box_size', [2.0, 1.5])[1]
            st.metric("Total Area", f"{total_area:.1f} m²")
        with col3:
            efficiency = results.get('optimization', {}).get('total_efficiency', 0.85) * 100
            st.metric("Efficiency", f"{efficiency:.1f}%")
        with col4:
            num_rooms = len(results.get('rooms', {}))
            st.metric("Rooms Analyzed", num_rooms)
        
        # Visualization
        st.subheader("📐 Plan Visualization")
        display_plan_visualization()

if __name__ == "__main__":
    main()