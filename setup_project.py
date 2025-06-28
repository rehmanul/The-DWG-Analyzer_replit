
#!/usr/bin/env python3
"""
Automated Project Setup Script
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing Python dependencies...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    print("✅ Python dependencies installed")

def create_streamlit_config():
    """Create Streamlit configuration if missing"""
    config_dir = Path('.streamlit')
    config_file = config_dir / 'config.toml'
    
    if not config_file.exists():
        print("⚙️ Creating Streamlit configuration...")
        config_dir.mkdir(exist_ok=True)
        
        config_content = """[server]
headless = true
address = "0.0.0.0"
port = 5000
maxUploadSize = 200

[theme]
base = "light"
primaryColor = "#1f77b4"

[browser]
gatherUsageStats = false
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        print("✅ Streamlit configuration created")

def setup_environment():
    """Setup environment variables"""
    if not os.environ.get('DATABASE_URL'):
        os.environ['DATABASE_URL'] = 'sqlite:///local_database.db'
        print("📊 Database URL set to SQLite (development mode)")
    
    if not os.environ.get('GEMINI_API_KEY'):
        print("🔑 Gemini API key not set - AI features will be limited")
        print("   To enable AI features, set GEMINI_API_KEY in Secrets")

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'uploads',
        'exports',
        'cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("📁 Created necessary directories")

def main():
    """Main setup process"""
    print("🚀 Setting up AI Architectural Space Analyzer PRO...")
    print("=" * 50)
    
    try:
        install_dependencies()
        create_streamlit_config()
        setup_environment()
        create_directories()
        
        print("\n" + "=" * 50)
        print("✅ Project setup completed successfully!")
        print("\n🎯 To start the application:")
        print("   python main.py")
        print("\n🔧 For development mode:")
        print("   streamlit run streamlit_app.py --server.port 5000 --server.address 0.0.0.0")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
