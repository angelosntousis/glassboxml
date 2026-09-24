"""Test the canvas event/state contract without requiring a browser or JS packages."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_canvas_events_preserve_strokes_and_clear_state():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is optional; needed only for the drawing event test")
    script = Path(__file__).parent / "frontend" / "drawing.test.mjs"
    result = subprocess.run(
        [node, "--test", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
