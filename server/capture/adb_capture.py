"""ADB screen capture module with wireless ADB support."""
import subprocess
import numpy as np
from typing import Optional
import cv2


class ADBCapture:
    """Captures screen from Android device via ADB (USB or wireless)."""
    
    def __init__(self, buffer_size: int = 1, device_serial: Optional[str] = None):
        """
        Initialize ADB capture.
        
        Args:
            buffer_size: Number of frames to buffer (currently unused, kept for API)
            device_serial: Device serial number or IP:port for wireless ADB
                          e.g., "192.168.1.50:5555"
        """
        self.buffer_size = buffer_size
        self._buffer: Optional[np.ndarray] = None
        self.device_serial = device_serial
    
    def _adb_cmd(self, *args) -> list:
        """Build ADB command with optional device serial."""
        cmd = ["adb"]
        if self.device_serial:
            cmd.extend(["-s", self.device_serial])
        cmd.extend(args)
        return cmd
    
    def connect(self) -> bool:
        """Connect to a device via wireless ADB.
        
        Returns:
            True if connection successful, False otherwise.
        """
        if not self.device_serial:
            print("No device serial set, assuming USB connection")
            return True
        
        try:
            result = subprocess.run(
                self._adb_cmd("connect", self.device_serial),
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout.strip()
            connected = "connected" in output.lower()
            if connected:
                print(f"Connected to {self.device_serial}")
            else:
                print(f"Connection failed: {output}")
            return connected
        except Exception as e:
            print(f"ADB connect error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the device."""
        if not self.device_serial:
            return
        try:
            subprocess.run(
                self._adb_cmd("disconnect", self.device_serial),
                capture_output=True, timeout=5
            )
            print(f"Disconnected from {self.device_serial}")
        except Exception:
            pass
    
    def is_connected(self) -> bool:
        """Check if the target device is connected."""
        try:
            result = subprocess.run(
                ["adb", "devices"],
                capture_output=True, text=True, timeout=5
            )
            return self.device_serial in result.stdout if self.device_serial else True
        except Exception:
            return False
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture a single frame from the device.
        
        Returns:
            numpy.ndarray: BGR image array or None if capture fails.
        """
        try:
            result = subprocess.run(
                self._adb_cmd("exec-out", "screencap", "-p"),
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return None
            
            # Convert PNG bytes to numpy array
            nparr = np.frombuffer(result.stdout, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"ADB capture error: {e}")
            return None
    
    def capture_loop(self, callback, fps: int = 10):
        """Run continuous capture loop.
        
        Args:
            callback: Function to call with each frame
            fps: Target frames per second
        """
        import time
        delay = 1.0 / fps
        
        while True:
            frame = self.capture()
            if frame is not None:
                callback(frame)
            time.sleep(delay)
