#!/usr/bin/env python3
"""
ADB Screenshot Capture Script for MLBB Drafter

Captures real draft screenshots from a connected Android device.
Use these screenshots to train the YOLOv8n-OBB detector.

Usage:
    python training/capture_screenshots.py --count 100 --output training/data/real/
    python training/capture_screenshots.py --count 50 --interval 2.0
    python training/capture_screenshots.py --list-devices

Requirements:
    - ADB installed and in PATH
    - Android device connected via USB with USB debugging enabled
    - MLBB app open to draft screen
"""

import argparse
import subprocess
import time
import os
from pathlib import Path
from datetime import datetime


def check_adb() -> bool:
    """Check if ADB is available."""
    try:
        result = subprocess.run(["adb", "version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def list_devices() -> list:
    """List connected ADB devices."""
    try:
        result = subprocess.run(["adb", "devices", "-l"], capture_output=True, text=True, timeout=5)
        devices = []
        for line in result.stdout.strip().split("\n")[1:]:
            if line.strip() and "device" in line:
                parts = line.split()
                devices.append({"serial": parts[0], "info": " ".join(parts[1:])})
        return devices
    except Exception as e:
        print(f"Error listing devices: {e}")
        return []


def capture_screenshot(output_path: Path, serial: str = None) -> bool:
    """Capture a single screenshot from the device."""
    try:
        cmd = ["adb"]
        if serial:
            cmd.extend(["-s", serial])
        cmd.extend(["exec-out", "screencap", "-p"])
        
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        
        if result.returncode != 0:
            print(f"ADB capture failed: {result.stderr.decode()}")
            return False
        
        if len(result.stdout) < 1000:  # Too small, probably an error
            print(f"Screenshot too small ({len(result.stdout)} bytes), likely invalid")
            return False
        
        with open(output_path, "wb") as f:
            f.write(result.stdout)
        
        return True
    except subprocess.TimeoutExpired:
        print("ADB capture timed out")
        return False
    except Exception as e:
        print(f"Capture error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Capture MLBB draft screenshots via ADB")
    parser.add_argument("--count", type=int, default=100, help="Number of screenshots to capture")
    parser.add_argument("--output", type=str, default="training/data/real", help="Output directory")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between captures")
    parser.add_argument("--serial", type=str, default=None, help="Device serial number (if multiple devices)")
    parser.add_argument("--list-devices", action="store_true", help="List connected devices and exit")
    parser.add_argument("--continuous", action="store_true", help="Capture continuously until interrupted")
    
    args = parser.parse_args()
    
    if args.list_devices:
        devices = list_devices()
        if not devices:
            print("No devices found. Check USB connection and USB debugging.")
        else:
            print(f"Found {len(devices)} device(s):")
            for d in devices:
                print(f"  {d['serial']}: {d['info']}")
        return
    
    if not check_adb():
        print("ERROR: ADB not found. Install Android SDK platform-tools.")
        print("  Ubuntu/Debian: sudo apt install adb")
        print("  macOS: brew install android-platform-tools")
        print("  Windows: Download from https://developer.android.com/studio/releases/platform-tools")
        return
    
    devices = list_devices()
    if not devices:
        print("No devices found. Connect your phone via USB and enable USB debugging.")
        return
    
    if args.serial is None and len(devices) > 1:
        print(f"Multiple devices found ({len(devices)}). Specify one with --serial:")
        for d in devices:
            print(f"  --serial {d['serial']}")
        return
    
    serial = args.serial or devices[0]["serial"]
    print(f"Using device: {serial}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Capturing {args.count} screenshots to {output_dir}/")
    print(f"Interval: {args.interval}s")
    print("Make sure MLBB is open to the draft screen!")
    print("Press Ctrl+C to stop early.\n")
    
    captured = 0
    failed = 0
    start_time = time.time()
    
    try:
        while captured < args.count or args.continuous:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"draft_{timestamp}_{captured:04d}.png"
            filepath = output_dir / filename
            
            if capture_screenshot(filepath, serial):
                captured += 1
                size_kb = filepath.stat().st_size / 1024
                elapsed = time.time() - start_time
                rate = captured / elapsed if elapsed > 0 else 0
                print(f"[{captured}/{args.count}] {filename} ({size_kb:.0f} KB) - {rate:.1f} fps")
            else:
                failed += 1
                if failed > 5:
                    print("Too many failures. Check device connection.")
                    break
            
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    elapsed = time.time() - start_time
    print(f"\nCapture complete:")
    print(f"  Captured: {captured} screenshots")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Average rate: {captured/elapsed:.2f} fps" if elapsed > 0 else "")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
