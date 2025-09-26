#!/usr/bin/env python3
"""
Health Check Script for LLM Bootcamp Chatbot
Diagnoses common issues and validates system components
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def print_status(message, status="INFO"):
    """Print formatted status message."""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"     # Reset
    }
    print(f"{colors.get(status, '')}{status}: {message}{colors['RESET']}")

def check_python_version():
    """Check Python version compatibility."""
    print_status("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - OK", "SUCCESS")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", "ERROR")
        return False

def check_virtual_environment():
    """Check if virtual environment is activated."""
    print_status("Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_status("Virtual environment is activated", "SUCCESS")
        return True
    else:
        print_status("Virtual environment not detected", "WARNING")
        return False

def check_required_packages():
    """Check if required packages are installed."""
    print_status("Checking required packages...")
    
    required_packages = [
        'streamlit',
        'langchain_openai', 
        'langchain_core',
        'openai',
        'tiktoken'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"✓ {package}", "SUCCESS")
        except ImportError:
            print_status(f"✗ {package} - MISSING", "ERROR")
            missing_packages.append(package)
    
    if missing_packages:
        print_status(f"Install missing packages: pip install {' '.join(missing_packages)}", "WARNING")
        return False
    
    return True

def check_project_structure():
    """Check project file structure."""
    print_status("Checking project structure...")
    
    required_files = [
        "Home.py",
        "pages/1_Basic_Chatbot.py", 
        "themes/modern_theme.py",
        "requirements.txt"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_status(f"✓ {file_path}", "SUCCESS")
        else:
            print_status(f"✗ {file_path} - MISSING", "ERROR")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_streamlit_process():
    """Check if Streamlit is running."""
    print_status("Checking Streamlit process...")
    
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'streamlit' in result.stdout:
            print_status("Streamlit process is running", "SUCCESS")
            
            # Extract port information
            lines = result.stdout.split('\n')
            for line in lines:
                if 'streamlit' in line and 'run' in line:
                    print_status(f"Process: {line.strip()}", "INFO")
            return True
        else:
            print_status("No Streamlit process found", "WARNING")
            return False
    except Exception as e:
        print_status(f"Error checking processes: {e}", "ERROR")
        return False

def check_port_availability():
    """Check if common ports are available."""
    print_status("Checking port availability...")
    
    import socket
    
    ports_to_check = [8501, 8502, 8503]
    
    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print_status(f"Port {port} - IN USE", "WARNING")
        else:
            print_status(f"Port {port} - AVAILABLE", "SUCCESS")

def check_api_key():
    """Check for OpenAI API key."""
    print_status("Checking API key configuration...")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        if api_key.startswith("sk-"):
            print_status("OpenAI API key found in environment", "SUCCESS")
        else:
            print_status("Invalid API key format", "ERROR")
        return True
    else:
        print_status("OpenAI API key not found in environment", "WARNING")
        print_status("Set with: export OPENAI_API_KEY=your_key_here", "INFO")
        return False

def test_basic_chatbot_import():
    """Test importing the Basic Chatbot module."""
    print_status("Testing Basic Chatbot import...")
    
    try:
        sys.path.append('.')
        spec = importlib.util.spec_from_file_location("basic_chatbot", "pages/1_Basic_Chatbot.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print_status("Basic Chatbot module loads successfully", "SUCCESS")
        return True
    except Exception as e:
        print_status(f"Error importing Basic Chatbot: {e}", "ERROR")
        return False

def suggest_fixes():
    """Provide suggestions for common issues."""
    print_status("=== COMMON FIXES ===", "INFO")
    print_status("1. Install dependencies: pip install -r requirements.txt", "INFO")
    print_status("2. Activate virtual environment: source venv/bin/activate", "INFO") 
    print_status("3. Set API key: export OPENAI_API_KEY=your_key_here", "INFO")
    print_status("4. Kill existing processes: pkill -f streamlit", "INFO")
    print_status("5. Restart Streamlit: streamlit run Home.py", "INFO")
    print_status("6. Check browser URL: http://localhost:8501", "INFO")

def main():
    """Run all health checks."""
    print_status("=== LLM BOOTCAMP CHATBOT HEALTH CHECK ===", "INFO")
    print_status(f"Working directory: {os.getcwd()}", "INFO")
    print("")
    
    checks = [
        check_python_version,
        check_virtual_environment, 
        check_required_packages,
        check_project_structure,
        check_streamlit_process,
        check_port_availability,
        check_api_key,
        test_basic_chatbot_import
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        if check():
            passed += 1
        print("")
    
    print_status("=== HEALTH CHECK SUMMARY ===", "INFO")
    print_status(f"Passed: {passed}/{total} checks", "SUCCESS" if passed == total else "WARNING")
    
    if passed < total:
        print("")
        suggest_fixes()
    else:
        print_status("All checks passed! Chatbot should be working properly.", "SUCCESS")
        print_status("Access at: http://localhost:8501", "INFO")

if __name__ == "__main__":
    main()