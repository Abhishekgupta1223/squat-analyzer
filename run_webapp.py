#!/usr/bin/env python3
"""
Launch Squat Analyzer Pro Web Application
=========================================

Usage:
    python run_webapp.py [--host HOST] [--port PORT] [--reload]
    
Examples:
    python run_webapp.py                    # Run on localhost:8000
    python run_webapp.py --port 3000        # Run on localhost:3000
    python run_webapp.py --host 0.0.0.0     # Allow external access
    python run_webapp.py --reload           # Auto-reload on code changes
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path
from time import sleep
from threading import Thread

# Ensure proper imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)


def open_browser(url: str, delay: float = 2.0):
    """Open browser after a delay."""
    sleep(delay)
    print(f"\n🌐 Opening {url} in your browser...")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(
        description="Launch Squat Analyzer Pro Web Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_webapp.py                    Run on localhost:8000
  python run_webapp.py --port 3000        Run on custom port
  python run_webapp.py --host 0.0.0.0     Allow network access
  python run_webapp.py --reload           Enable hot reload
  python run_webapp.py --no-browser       Don't open browser
        """
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    # Banner
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   🏋️  SQUAT ANALYZER PRO - Web Application                        ║
║                                                                   ║
║   Real-time AI-powered squat form analysis                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
    print(f"🚀 Starting server at {url}")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print("⏹️  Press Ctrl+C to stop\n")
    
    # Open browser in background thread
    if not args.no_browser:
        Thread(target=open_browser, args=(url,), daemon=True).start()
    
    # Import and run uvicorn
    import uvicorn
    
    # Import the app from webapp.server
    from webapp.server import app
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
