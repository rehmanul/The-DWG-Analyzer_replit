
#!/usr/bin/env python3
"""
Project Status Checker and Health Report Generator
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import json
from datetime import datetime

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (Requires 3.8+)"

def check_dependencies():
    """Check all project dependencies"""
    requirements_file = Path('requirements.txt')
    if not requirements_file.exists():
        return False, "requirements.txt not found"
    
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    installed = []
    missing = []
    
    for req in requirements:
        package_name = req.split('>=')[0].split('==')[0].split('>')[0].split('<')[0]
        try:
            importlib.import_module(package_name.replace('-', '_'))
            installed.append(package_name)
        except ImportError:
            missing.append(package_name)
    
    return len(missing) == 0, {"installed": installed, "missing": missing}

def check_file_structure():
    """Check essential project files"""
    essential_files = [
        'main.py',
        'streamlit_app.py',
        'requirements.txt',
        'src/__init__.py',
        'src/ai_integration.py',
        'src/dwg_parser.py',
        'src/navigation_manager.py'
    ]
    
    existing = []
    missing = []
    
    for file_path in essential_files:
        if Path(file_path).exists():
            existing.append(file_path)
        else:
            missing.append(file_path)
    
    return len(missing) == 0, {"existing": existing, "missing": missing}

def check_environment_variables():
    """Check important environment variables"""
    env_vars = {
        'DATABASE_URL': os.environ.get('DATABASE_URL'),
        'GEMINI_API_KEY': bool(os.environ.get('GEMINI_API_KEY')),
        'PORT': os.environ.get('PORT', '5000')
    }
    return True, env_vars

def check_streamlit_config():
    """Check Streamlit configuration"""
    config_path = Path('.streamlit/config.toml')
    if config_path.exists():
        return True, "Streamlit config found"
    return False, "Streamlit config missing"

def run_basic_import_test():
    """Test basic imports of main modules"""
    test_imports = [
        'streamlit',
        'src.ai_integration',
        'src.dwg_parser',
        'src.navigation_manager'
    ]
    
    successful = []
    failed = []
    
    for module in test_imports:
        try:
            importlib.import_module(module)
            successful.append(module)
        except Exception as e:
            failed.append(f"{module}: {str(e)}")
    
    return len(failed) == 0, {"successful": successful, "failed": failed}

def generate_status_report():
    """Generate comprehensive status report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "project": "AI Architectural Space Analyzer PRO",
        "checks": {}
    }
    
    # Run all checks
    checks = [
        ("python_version", check_python_version),
        ("dependencies", check_dependencies),
        ("file_structure", check_file_structure),
        ("environment_variables", check_environment_variables),
        ("streamlit_config", check_streamlit_config),
        ("import_test", run_basic_import_test)
    ]
    
    overall_status = True
    
    for check_name, check_function in checks:
        try:
            success, details = check_function()
            report["checks"][check_name] = {
                "status": "PASS" if success else "FAIL",
                "details": details
            }
            if not success:
                overall_status = False
        except Exception as e:
            report["checks"][check_name] = {
                "status": "ERROR",
                "details": str(e)
            }
            overall_status = False
    
    report["overall_status"] = "HEALTHY" if overall_status else "ISSUES_FOUND"
    
    return report

def print_report(report):
    """Print formatted report to console"""
    print("=" * 60)
    print(f"PROJECT STATUS REPORT - {report['project']}")
    print(f"Generated: {report['timestamp']}")
    print("=" * 60)
    
    print(f"\nOVERALL STATUS: {report['overall_status']}")
    print("-" * 40)
    
    for check_name, check_data in report["checks"].items():
        status = check_data["status"]
        print(f"\n{check_name.upper().replace('_', ' ')}: {status}")
        
        if isinstance(check_data["details"], dict):
            for key, value in check_data["details"].items():
                print(f"  {key}: {value}")
        else:
            print(f"  {check_data['details']}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    
    if report["overall_status"] == "HEALTHY":
        print("✅ Project is ready to run!")
        print("   Use: python main.py")
    else:
        print("⚠️  Issues found. Please address:")
        for check_name, check_data in report["checks"].items():
            if check_data["status"] != "PASS":
                if check_name == "dependencies" and "missing" in str(check_data["details"]):
                    print(f"   - Install missing packages: pip install -r requirements.txt")
                elif check_name == "file_structure":
                    print(f"   - Missing files: {check_data['details'].get('missing', [])}")
                elif check_name == "python_version":
                    print(f"   - Upgrade Python to 3.8 or higher")

def main():
    """Main execution"""
    report = generate_status_report()
    
    # Print to console
    print_report(report)
    
    # Save to file
    with open('project_status_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: project_status_report.json")
    
    # Return exit code based on status
    return 0 if report["overall_status"] == "HEALTHY" else 1

if __name__ == "__main__":
    sys.exit(main())
