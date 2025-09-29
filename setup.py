#!/usr/bin/env python3
"""
Setup script for WaveNet Audio Generation Project
This script helps set up the project environment
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    try:
        import torch
        import torchaudio
        import numpy
        import matplotlib
        import scipy
        import librosa
        import streamlit
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ["outputs", "logs", "checkpoints"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  Created: {directory}")
        else:
            print(f"  Exists: {directory}")

def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "test_wavenet.py"])
        print("✅ All tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🎵 WaveNet Audio Generation - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but setup completed")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python demo.py' to see the project in action")
    print("2. Run 'python 0137.py' for full training and generation")
    print("3. Run 'streamlit run app.py' for the web interface")
    print("4. Check README.md for detailed documentation")

if __name__ == "__main__":
    main()
