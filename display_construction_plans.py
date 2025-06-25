"""
Construction Plans Display Function
"""
import streamlit as st

def display_construction_plans(components):
    """Display construction plans with 2D/3D visualization"""
    if not st.session_state.zones:
        st.info("Load a DWG file to see construction plans")
        return
    
    construction_planner = components.get('construction_planner')
    if not construction_planner:
        st.error("Construction planner not available")
        return
    
    st.subheader("🏗️ Construction Plans & Visualization")
    
    # Create empty plan from zones
    plan = construction_planner.create_empty_plan(st.session_state.zones)
    
    # Plan options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        plan_type = st.selectbox("Plan Type", ["Empty Plan", "With Furniture"])
    with col2:
        view_mode = st.selectbox("View Mode", ["2D Plan", "3D Model"])
    with col3:
        show_details = st.checkbox("Show Details", value=True)
    
    # Set analysis results if available
    if st.session_state.analysis_results:
        construction_planner.set_analysis_results(st.session_state.analysis_results)
    
    # Display visualization
    if view_mode == "2D Plan":
        show_boxes = (plan_type == "With Furniture")
        fig_2d = construction_planner.create_2d_plan_visualization(plan, show_boxes=show_boxes)
        st.plotly_chart(fig_2d, use_container_width=True)
    else:
        show_boxes = (plan_type == "With Furniture")
        fig_3d = construction_planner.create_3d_plan_visualization(plan, show_boxes=show_boxes)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Construction report
    if show_details:
        st.divider()
        st.subheader("📋 Construction Report")
        
        report = construction_planner.generate_construction_report(plan)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rooms", report['total_rooms'])
        with col2:
            st.metric("Total Area", f"{report['total_area']:.1f} m²")
        with col3:
            total_doors = sum(len(zone.get('doors', [])) for zone in plan['zones'])
            st.metric("Total Doors", total_doors)
        with col4:
            total_windows = sum(len(zone.get('windows', [])) for zone in plan['zones'])
            st.metric("Total Windows", total_windows)
        
        # Room details
        st.subheader("Room Details")
        room_data = []
        for room in report['room_details']:
            room_data.append({
                'Room': room['name'],
                'Area (m²)': f"{room['area']:.1f}",
                'Perimeter (m)': f"{room['perimeter']:.1f}",
                'Doors': room['doors'],
                'Windows': room['windows'],
                'Wall Length (m)': f"{room['wall_length']:.1f}"
            })
        
        import pandas as pd
        df = pd.DataFrame(room_data)
        st.dataframe(df, use_container_width=True)
        
        # Materials needed
        st.subheader("Materials Estimate")
        materials = report['materials_needed']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Structural Materials:**")
            st.write(f"• Concrete blocks: {materials['concrete_blocks']} units")
            st.write(f"• Cement bags: {materials['cement_bags']} bags")
            st.write(f"• Steel bars: {materials['steel_bars']} units")
        
        with col2:
            st.write("**Openings:**")
            st.write(f"• Doors: {materials['doors']} units")
            st.write(f"• Windows: {materials['windows']} units")