#!/usr/bin/env python3
"""
Auto Preview - Antigravity Kit
==============================
Manages (start/stop/status) the local development server for previewing the application.

Usage:
    python .agent/scripts/auto_preview.py start [port]
    python .agent/scripts/auto_preview.py stop
    python .agent/scripts/auto_preview.py status
"""

import os
import sys
import time
import json
import signal
import argparse
import subprocess
from pathlib import Path

# Configure console encoding to UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

AGENT_DIR = Path(".agent")
PID_FILE = AGENT_DIR / "preview.pid"
LOG_FILE = AGENT_DIR / "preview.log"
API_LOG_FILE = AGENT_DIR / "preview_api.log"
WEB_LOG_FILE = AGENT_DIR / "preview_web.log"


def get_project_root():
    return Path(".").resolve()


def is_running(pid):
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def get_project_type(root):
    if (root / "package.json").exists():
        return "node"
    elif (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
        return "python"
    return None


def get_start_command(root):
    pkg_file = root / "package.json"
    if not pkg_file.exists():
        return None

    with open(pkg_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scripts = data.get("scripts", {})
    if "dev" in scripts:
        return ["npm", "run", "dev"]
    elif "start" in scripts:
        return ["npm", "start"]
    return None


def start_server(port=3000):
    if PID_FILE.exists():
        try:
            pids = PID_FILE.read_text().strip().split()
            running_pids = []
            for pid_str in pids:
                pid = int(pid_str)
                if is_running(pid):
                    running_pids.append(pid)
            if running_pids:
                print(f"⚠️  Preview already running (PIDs: {', '.join(map(str, running_pids))})")
                return
        except Exception:
            pass  # Invalid PID file

    root = get_project_root()
    project_type = get_project_type(root)

    if not project_type:
        print("❌ Could not identify project type (no package.json or pyproject.toml found)")
        sys.exit(1)

    env = os.environ.copy()
    pids = []

    if project_type == "node":
        cmd = get_start_command(root)
        if not cmd:
            print("❌ No 'dev' or 'start' script found in package.json")
            sys.exit(1)

        env["PORT"] = str(port)
        print(f"🚀 Starting Node.js preview on port {port}...")

        with open(LOG_FILE, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                cmd, cwd=str(root), stdout=log, stderr=log, env=env, shell=True
            )
        pids.append(process.pid)
        print(f"✅ Preview started! (PID: {process.pid})")
        print(f"   Logs: {LOG_FILE}")
        print(f"   URL: http://localhost:{port}")

    elif project_type == "python":
        env["PYTHONPATH"] = str(root)

        print(f"🚀 Starting FastAPI app (port {port})...")
        api_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]
        with open(API_LOG_FILE, "w", encoding="utf-8") as log:
            api_process = subprocess.Popen(api_cmd, cwd=str(root), stdout=log, stderr=log, env=env)
        pids.append(api_process.pid)

        print(f"✅ Preview started!")
        print(f"   URL:  http://localhost:{port}")
        print(f"   Logs: {API_LOG_FILE}")

    PID_FILE.write_text(" ".join(map(str, pids)))


def stop_server():
    if not PID_FILE.exists():
        print("ℹ️  No preview server found.")
        return

    try:
        pids_str = PID_FILE.read_text().strip()
        if not pids_str:
            print("ℹ️  No preview server running (empty PID file).")
            return

        pids = [int(p) for p in pids_str.split()]
        stopped_count = 0
        for pid in pids:
            if is_running(pid):
                if sys.platform != "win32":
                    os.kill(pid, signal.SIGTERM)
                else:
                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                print(f"🛑 Stopped process (PID: {pid})")
                stopped_count += 1
        if stopped_count == 0:
            print("ℹ️  Processes were not running.")
    except Exception as e:
        print(f"❌ Error stopping server: {e}")
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()


def status_server():
    running_pids = []

    if PID_FILE.exists():
        try:
            pids = [int(p) for p in PID_FILE.read_text().strip().split()]
            for pid in pids:
                if is_running(pid):
                    running_pids.append(pid)
        except Exception:
            pass

    print("\n=== Preview Status ===")
    if running_pids:
        print("✅ Status: Running")
        print(f"🔢 PIDs: {', '.join(map(str, running_pids))}")

        root = get_project_root()
        project_type = get_project_type(root)
        if project_type == "python":
            print("🌐 URL: http://localhost:3000 (Likely)")
            print(f"📝 Logs: {API_LOG_FILE}")
        else:
            print("🌐 URL: http://localhost:3000 (Likely)")
            print(f"📝 Logs: {LOG_FILE}")
    else:
        print("⚪ Status: Stopped")
    print("===================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("port", nargs="?", default="3000")

    args = parser.parse_args()

    if args.action == "start":
        start_server(int(args.port))
    elif args.action == "stop":
        stop_server()
    elif args.action == "status":
        status_server()


if __name__ == "__main__":
    main()
