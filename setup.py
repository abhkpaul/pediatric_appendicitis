#!/usr/bin/env python3
"""
Setup script for CPU-only environment.
"""

import os
import subprocess
import sys


def setup_environment():
    print("Setting up CPU-only environment...")

    # Create directories
    directories = [
        'data/raw',
        'data/processed',
        'data/features',
        'models',
        'results',
        'configs',
        'src',
        'notebooks',
        'scripts',
        'tests'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Install requirements
    print("Installing requirements...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

    # Download spaCy model
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])

    print("✅ Environment setup completed!")


if __name__ == "__main__":
    setup_environment()