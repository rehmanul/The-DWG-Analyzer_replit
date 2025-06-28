
#!/usr/bin/env python3
"""
AI Architectural Space Analyzer PRO
Main application entry point
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    # Map package names to their import names
    package_imports = {
        'streamlit': 'streamlit',
        'ezdxf': 'ezdxf',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'shapely': 'shapely',
        'scikit-learn': 'sklearn'
    }
    
    missing_packages = []
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("All required packages should be available via pyproject.toml dependencies")
        logger.info("If packages are missing, they should be installed automatically by the Nix environment")
        # Don't try to install via pip in Nix environment
        return False
    
    logger.info("All required dependencies are available")
    return True

def run_streamlit_app():
    """Run the main Streamlit application"""
    try:
        # Check if we should use streamlit_app.py or app.py
        if Path('streamlit_app.py').exists():
            app_file = 'streamlit_app.py'
        elif Path('app.py').exists():
            app_file = 'app.py'
        else:
            raise FileNotFoundError("No main application file found (streamlit_app.py or app.py)")
        
        logger.info(f"Starting Streamlit application: {app_file}")
        
        # Run Streamlit with proper configuration
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', app_file,
            '--server.port', '5000',
            '--server.address', '0.0.0.0',
            '--server.maxUploadSize', '200',
            '--server.headless', 'true'
        ]
        
        subprocess.run(cmd)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    logger.info("AI Architectural Space Analyzer PRO - Starting...")
    
    # Check dependencies - continue even if some are missing
    dependencies_ok = check_dependencies()
    if not dependencies_ok:
        logger.warning("Some dependencies may be missing, but continuing anyway...")
    
    # Set environment variables if not set
    if not os.environ.get('DATABASE_URL'):
        os.environ['DATABASE_URL'] = 'sqlite:///local_database.db'
        logger.info("Using SQLite database (development mode)")
    
    # Run the application
    run_streamlit_app()

if __name__ == "__main__":
    main()
