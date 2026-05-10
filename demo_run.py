#!/usr/bin/env python3
"""
Launch the interactive ambigram demo.

  python demo_run.py

Opens http://localhost:8000 in your browser automatically.
Type a word, press Generate, and watch the SVG ambigram form in real time.

Prerequisites (one-time):
  python tools/generate_font_dataset.py
  python tools/train_classifier.py
"""
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def main() -> None:
    ckpt = Path("data/char_classifier.pth")
    if not ckpt.exists():
        print("Character classifier not found. Run these first:\n")
        print("  python tools/generate_font_dataset.py")
        print("  python tools/train_classifier.py\n")
        sys.exit(1)

    url = "http://localhost:8000"
    print(f"\n  Ambigram demo  →  {url}\n")
    print("  Press Ctrl-C to stop.\n")

    def _open_browser() -> None:
        time.sleep(1.8)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()

    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "demo.server:app",
                "--host", "127.0.0.1",
                "--port", "8000",
                "--log-level", "warning",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
