
#!/usr/bin/env python3
"""
Simple test script to verify all dependencies are working
"""

import sys

def test_imports():
    """Test all required imports"""
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
        
        import ezdxf
        print("✓ ezdxf imported successfully")
        
        import plotly
        print("✓ Plotly imported successfully")
        
        import pandas as pd
        print("✓ Pandas imported successfully")
        
        import numpy as np
        print("✓ NumPy imported successfully")
        
        import matplotlib
        print("✓ Matplotlib imported successfully")
        
        import shapely
        print("✓ Shapely imported successfully")
        
        import sklearn
        print("✓ Scikit-learn imported successfully")
        
        print("\n✅ All core dependencies are working!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    if test_imports():
        print("\n🚀 Ready to run the application!")
        sys.exit(0)
    else:
        print("\n❌ Some dependencies are missing")
        sys.exit(1)
