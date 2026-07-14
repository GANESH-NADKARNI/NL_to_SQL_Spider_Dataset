"""
quick_train.py — Train only Rule-Based + Random Forest (fast, no GPU needed)
Use this to verify the setup works before running full training.

Usage: python quick_train.py
"""
import subprocess, sys

cmd = [sys.executable, "train.py", "--models", "rule", "rf"]
print("Running quick training (Rule-Based + Random Forest only)...")
print("This takes ~5-10 minutes on CPU.\n")
subprocess.run(cmd, check=True)
