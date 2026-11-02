"""Offline tests for the UDP control input (control_in.py): a busy port must not crash the app.

Uses ephemeral ports on 127.0.0.1 only. Run from the repository root:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import io
import os
import socket
import sys
import time
import unittest
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquarium_boids.control_in import ControlReceiver  # noqa: E402
from aquarium_boids.controls import RuntimeControls  # noqa: E402


def busy_port() -> tuple[socket.socket, int]:
    """A UDP socket holding a free port; the caller closes it."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    holder.bind(("127.0.0.1", 0))
    return holder, holder.getsockname()[1]


class BusyPort(unittest.TestCase):
    def test_start_does_not_raise_and_stays_idle(self):
        holder, port = busy_port()
        try:
            receiver = ControlReceiver(port=port)
            receiver.start()  # used to raise OSError (WinError 10048)
            self.assertIsNotNone(receiver.error)
            self.assertEqual(receiver.drain(RuntimeControls()), [])
            receiver.stop()  # safe without a socket
        finally:
            holder.close()

    def test_main_prints_a_warning_and_continues(self):
        import main  # src/main.py

        holder, port = busy_port()
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                receiver = main.start_control_input(port)
        finally:
            holder.close()
        self.assertIn(f"WARNING: cannot listen on UDP {port}", out.getvalue())
        self.assertIn("Continuing without control input", out.getvalue())
        self.assertNotIn("Listening for", out.getvalue())
        self.assertEqual(receiver.drain(RuntimeControls()), [])


class FreePort(unittest.TestCase):
    def test_receives_and_applies_controls(self):
        holder, port = busy_port()
        holder.close()  # now free
        out = io.StringIO()
        import main

        with contextlib.redirect_stdout(out):
            receiver = main.start_control_input(port)
        try:
            self.assertIsNone(receiver.error)
            self.assertIn(f"Listening for Max/MIDIMIX controls on UDP {port}", out.getvalue())
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sender.sendto(b"control cohesion_weight 2.2;", ("127.0.0.1", port))
            sender.close()
            controls = RuntimeControls()
            applied = []
            deadline = time.time() + 3.0
            while not applied and time.time() < deadline:
                applied = receiver.drain(controls)
                time.sleep(0.02)
            self.assertEqual([c.name for c in applied], ["cohesion_weight"])
            self.assertAlmostEqual(controls.cohesion_weight, 2.2)
        finally:
            receiver.stop()


if __name__ == "__main__":
    unittest.main()
