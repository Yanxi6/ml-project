import subprocess
import sys


def test_train_script_runs():
    # Just checks the script can run without crashing.
    result = subprocess.run([sys.executable, "src/train.py"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
