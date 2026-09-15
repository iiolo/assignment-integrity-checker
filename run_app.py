"""Entry point for the standalone .exe build: launches the Streamlit server directly."""
import os
import sys
import threading
import time
import webbrowser

from streamlit.web import cli as stcli


def resource_path(relative_path: str) -> str:
    # Resolve a path that works both in dev and inside a PyInstaller-frozen exe
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def open_browser_when_ready() -> None:
    # Headless mode skips streamlit's own onboarding-email prompt, so open the browser ourselves
    time.sleep(3)
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    sys.argv = [
        "streamlit",
        "run",
        resource_path("streamlit_app.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
    ]
    sys.exit(stcli.main())
