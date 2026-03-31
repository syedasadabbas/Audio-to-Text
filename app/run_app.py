"""
Launcher for the tkinter GUI application.

Run from anywhere:
  python app/run_app.py
  python -m app.run_app
"""
import sys
from pathlib import Path

# Ensure the project root (parent of app/) is on the path so that
# both the pipeline modules and the app package are importable.
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.app import TranscriptionApp

if __name__ == "__main__":
    TranscriptionApp().mainloop()
