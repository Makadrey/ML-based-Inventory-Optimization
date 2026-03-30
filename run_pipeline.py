# ══════════════════════════════════════════════════════════════════════════════
# run_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════
import subprocess
import sys
import time
from pathlib import Path

scripts = [
    "load_data.py",
    "preprocessing.py",
    "visualization.py",
    "analysis.py",
    "feature_engineering.py",
    "modelling.py",
    "lstm_model.py",
    "model_comparison.py"
]

scripts_dir = Path(__file__).parent

print("🚀 Starting Full Pipeline...\n" + "=" * 50)
for script in scripts:
    path = scripts_dir / script
    print(f"\n▶  Running {script}...")
    start  = time.time()
    result = subprocess.run([sys.executable, str(path)], capture_output=False)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"   ✅ {script} completed in {elapsed:.1f}s")
    else:
        print(f"   ❌ {script} FAILED (exit code {result.returncode})")
        print("   Fix the error above and re-run.")
        sys.exit(1)

print("\n" + "=" * 50)
print("✅ FULL PIPELINE COMPLETE!")
print("=" * 50)