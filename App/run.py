import subprocess
import sys
import os

def run_app():
    """
    Launch the Streamlit application.
    """
    try:
        print("🚀 Starting PlantVision AI...")
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install requirements first: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n👋 App closed.")

if __name__ == "__main__":
    run_app()
