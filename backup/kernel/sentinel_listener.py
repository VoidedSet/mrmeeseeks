import os
import json
import asyncio
import logging

from core import profiler_emitter

log = logging.getLogger("sentinel")

class SentinelListener:
    def __init__(self, brain_ref=None):
        self.brain_ref = brain_ref
        self.socket_path = "/tmp/meeseeks_sentinel.sock"
        self.sentinel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sentinel/build/sentinel-daemon"))
        self.server = None
        self.process = None
        self.keep_running = False
        self.active_connections = []
        self.supervisor_task = None

    async def start(self):
        self.keep_running = True
        log.info("Starting Sentinel Observability Listener...")
        
        # 1. Clean up stale socket file
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except Exception as e:
                log.error(f"Failed to delete stale socket {self.socket_path}: {e}")

        # 2. Start UDS Server
        try:
            self.server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)
            log.info(f"UDS Server listening on {self.socket_path}")
        except Exception as e:
            log.error(f"Failed to start UDS Server: {e}")
            return

        # 3. Start supervisor loop in background
        self.supervisor_task = asyncio.create_task(self._supervisor_loop())

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        log.info("[SentinelListener] Client connected to socket.")
        self.active_connections.append(writer)
        try:
            while self.keep_running:
                line = await reader.readline()
                if not line:
                    break
                
                try:
                    payload = json.loads(line.decode("utf-8").strip())
                    profiler_emitter.emit("sentinel_event", payload=payload)
                    # Dispatch to brain
                    if self.brain_ref is not None:
                        # Schedule task so it doesn't block the socket loop
                        asyncio.create_task(self.brain_ref.handle_proactive_event(payload))
                    else:
                        log.info(f"[Sentinel Event] {payload}")
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse event JSON: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Socket connection error: {e}")
        finally:
            log.info("[SentinelListener] Client disconnected.")
            if writer in self.active_connections:
                self.active_connections.remove(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _supervisor_loop(self):
        """Monitors and restarts the C++ sentinel-daemon process if it exits."""
        while self.keep_running:
            if not os.path.exists(self.sentinel_path):
                log.error(f"Sentinel daemon binary not found at {self.sentinel_path}. Build it first!")
                await asyncio.sleep(5)
                continue

            log.info(f"Spawning Sentinel daemon child process: {self.sentinel_path}")
            try:
                # Pass New Volume mount check path if any (Meeseeks knows about it)
                new_vol = "/media/kshayik/New Volume"
                self.process = await asyncio.create_subprocess_exec(
                    self.sentinel_path, new_vol,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # Read stdout/stderr logs in separate tasks
                stdout_task = asyncio.create_task(self._read_stream(self.process.stdout, "STDOUT"))
                stderr_task = asyncio.create_task(self._read_stream(self.process.stderr, "STDERR"))

                # Wait for process exit
                exit_code = await self.process.wait()
                log.warning(f"Sentinel daemon process exited with code {exit_code}")

                # Cancel log readers
                stdout_task.cancel()
                stderr_task.cancel()

            except Exception as e:
                log.error(f"Failed to spawn Sentinel process: {e}")

            self.process = None

            if self.keep_running:
                log.info("Restarting Sentinel daemon in 2 seconds (Supervisor loop)...")
                await asyncio.sleep(2)

    async def _read_stream(self, stream: asyncio.StreamReader, prefix: str):
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                log.info(f"[Daemon {prefix}] {line.decode('utf-8').strip()}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Error reading daemon {prefix}: {e}")

    async def stop(self):
        self.keep_running = False
        log.info("Stopping Sentinel Observability Listener...")
        
        # 1. Terminate supervisor task
        if self.supervisor_task:
            self.supervisor_task.cancel()
            try:
                await self.supervisor_task
            except asyncio.CancelledError:
                pass
            self.supervisor_task = None

        # 2. Terminate supervisor process
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception as e:
                log.error(f"Error terminating Sentinel process: {e}")
            self.process = None

        # 3. Close active socket connections
        for conn in list(self.active_connections):
            try:
                conn.close()
                await conn.wait_closed()
            except Exception:
                pass
        self.active_connections.clear()

        # 4. Stop socket server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        # 5. Clean up socket file
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except Exception:
                pass
        log.info("Sentinel Observability Listener stopped ✓")
