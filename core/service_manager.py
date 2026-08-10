import os
import sys
import socket
import asyncio
import logging
import shutil

log = logging.getLogger("service_manager")

class ServiceManager:
    def __init__(self):
        self.backend_proc = None
        self.backend_port = 11400

    def _is_port_open(self, port: int) -> bool:
        """Checks if a local port is already open and listening."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except (socket.timeout, ConnectionRefusedError):
                return False

    async def _wait_for_port(self, port: int, timeout: float = 15.0) -> bool:
        """Waits asynchronously for a port to start listening."""
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            if self._is_port_open(port):
                return True
            await asyncio.sleep(0.1)
        return False

    async def start_services(self):
        """Starts the unified Python backend if it is not already running."""
        log.info("Checking background services...")

        # 2. Start Unified Backend if not running
        if self._is_port_open(self.backend_port):
            log.info(f"Unified backend is already running on port {self.backend_port}.")
        else:
            backend_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "meeseeks_service.py"))
            log.info(f"Starting unified python backend: {backend_script}")
            try:
                self.backend_proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-u", backend_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                asyncio.create_task(self._log_stream(self.backend_proc.stdout, "Backend"))
                asyncio.create_task(self._log_stream(self.backend_proc.stderr, "Backend-Err"))
            except Exception as e:
                log.error(f"Failed to start unified python backend: {e}")

        # 3. Wait for backend to be fully up and listening
        log.info("Waiting for background services to wake up...")
        backend_ready = await self._wait_for_port(self.backend_port, timeout=10.0)

        if backend_ready:
            log.info("Unified Python backend is ready ✓")
        else:
            log.warning("Unified Python backend failed to start within timeout.")

    async def _log_stream(self, stream: asyncio.StreamReader, prefix: str):
        """Helper to read process output and log it."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
                if line_str:
                    log.debug(f"[{prefix}] {line_str}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Error reading stream from {prefix}: {e}")

    async def stop_services(self):
        """Cleanly stops any background services spawned by this instance."""
        log.info("Stopping background services...")
        
        # Stop Unified Backend
        if self.backend_proc:
            log.info("Terminating unified python backend...")
            try:
                self.backend_proc.terminate()
                await self.backend_proc.wait()
            except Exception as e:
                log.error(f"Error terminating unified backend: {e}")
            self.backend_proc = None
            
        log.info("Background services stopped ✓")

    def stop_services_sync(self):
        """Forcefully and synchronously terminates background services (safe for signal handlers)."""
        log.info("Stopping background services (sync)...")
        
        # Terminate Unified Python backend
        if self.backend_proc:
            try:
                self.backend_proc.kill()
            except Exception:
                pass
            self.backend_proc = None

service_manager = ServiceManager()
