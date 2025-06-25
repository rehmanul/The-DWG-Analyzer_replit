def load_uploaded_file(uploaded_file):
    """Load uploaded file with instant demo layouts"""
    if uploaded_file is None:
        st.error("No file provided")
        return None

    try:
        # Validate file first
        if not uploaded_file.name:
            st.error("Invalid file: No filename")
            return None

        # Check file extension
        file_ext = uploaded_file.name.lower().split('.')[-1]
        if file_ext not in ['dwg', 'dxf']:
            st.error(f"Unsupported file format: {file_ext}. Please upload DWG or DXF files only.")
            return None

        # Check file size
        try:
            file_bytes = uploaded_file.getvalue()
        except Exception as e:
            st.error(f"Could not read file: {str(e)}")
            return None

        if not file_bytes or len(file_bytes) == 0:
            st.error("File appears to be empty")
            return None

        file_size_mb = len(file_bytes) / (1024 * 1024)

        # Create instant demo layout based on filename
        zones = RobustErrorHandler.create_default_zones(uploaded_file.name, f"Demo layout for {uploaded_file.name}")
        st.success(f"✅ Instant layout ready: {len(zones)} zones for {uploaded_file.name}")

        # Update session state
        st.session_state.zones = zones
        st.session_state.file_loaded = True
        st.session_state.current_file = uploaded_file.name
        st.session_state.dwg_loaded = True
        st.session_state.analysis_results = {}
        st.session_state.analysis_complete = False

        return zones

    except Exception as e:
        st.error(f"File upload error: {str(e)}")
        # Emergency fallback
        zones = RobustErrorHandler.create_default_zones("emergency_fallback", "Error recovery")
        st.session_state.zones = zones
        st.session_state.file_loaded = True
        st.session_state.current_file = "fallback_file"
        st.session_state.dwg_loaded = True
        st.session_state.analysis_results = {}
        st.session_state.analysis_complete = False
        
        st.warning(f"Created emergency fallback layout with {len(zones)} zones")
        return zones