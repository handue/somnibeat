import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_are_rendered_from_current_results():
    """README.md and docs/WRITEUP.md must equal a fresh render from results/*.json."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts/render_docs.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
