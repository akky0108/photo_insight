import subprocess
import os
import sys


def run_pip_list(output_file):
    try:
        print("\n📦 Running 'pip list'...")
        result = subprocess.run(
            ["pip", "list", "--format=json"], check=True, capture_output=True, text=True
        )
        with open(output_file, "w") as f:
            f.write(result.stdout)
        print(f"✅ Generated {output_file}")
    except subprocess.CalledProcessError:
        print("❌ Failed to run pip list.")
        sys.exit(1)


def load_exclude_list(file_path):
    try:
        with open(file_path, "r") as f:
            return set(pkg.strip().lower() for pkg in f if pkg.strip())
    except Exception as e:
        print(f"❌ Failed to read exclude list: {e}")
        return set()
