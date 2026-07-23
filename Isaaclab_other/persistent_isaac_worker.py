"""Lifecycle management for one persistent Isaac Sim process per GPU slot."""

import json
import os
import signal
import subprocess
import time


class PersistentIsaacWorkerError(RuntimeError):
    """Raised when a persistent Isaac worker cannot serve a request."""


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


class PersistentIsaacWorker:
    """Run sequential training requests without restarting Isaac Sim per task."""

    def __init__(
        self,
        *,
        slot_id,
        device_name,
        request_dir,
        worker_script,
        python_executable,
        python_override_root,
        isaaclab_root,
        isaac_sim_setup,
        evolution_log_root,
        tmp_root,
        startup_timeout=600,
        request_timeout=7200,
        stall_timeout=600,
        max_requests=3,
        request_retries=1,
    ):
        self.slot_id = slot_id
        self.device_name = device_name
        self.request_dir = request_dir
        self.worker_script = worker_script
        self.python_executable = python_executable
        self.python_override_root = python_override_root
        self.isaaclab_root = isaaclab_root
        self.isaac_sim_setup = isaac_sim_setup
        self.evolution_log_root = evolution_log_root
        self.tmp_root = tmp_root
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.stall_timeout = stall_timeout
        self.max_requests = max_requests
        self.request_retries = request_retries
        self.process = None
        self._log_file = None
        self.worker_log = os.path.join(os.path.dirname(self.request_dir), "isaac_worker.log")
        self.requests_served = 0

    def _close_log_file(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _force_terminate(self):
        """Terminate the whole Isaac process group when a request stops progressing."""
        process = self.process
        if process is None or process.poll() is not None:
            self.process = None
            self._close_log_file()
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        finally:
            self.process = None
            self._close_log_file()

    def _remove_request_files(self, request_id):
        for suffix in ("request.json", "working.json", "response.json"):
            path = os.path.join(self.request_dir, f"{request_id}.{suffix}")
            if os.path.exists(path):
                os.remove(path)

    def _log_mtime(self):
        try:
            return os.path.getmtime(self.worker_log)
        except FileNotFoundError:
            return 0.0

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        if not os.path.exists(self.worker_script):
            raise PersistentIsaacWorkerError(f"Worker script is missing: {self.worker_script}")

        os.makedirs(self.request_dir, exist_ok=True)
        for name in os.listdir(self.request_dir):
            if name.endswith((".request.json", ".working.json", ".response.json", "ready.json")):
                os.remove(os.path.join(self.request_dir, name))
        os.makedirs(self.tmp_root, exist_ok=True)

        shell_cmd = f"""
        export TERM=xterm
        export PYTHONUNBUFFERED=1
        export ACCEPT_EULA=Y
        export CONDA_PREFIX=\"{os.path.dirname(os.path.dirname(self.python_executable))}\"
        export PATH=\"{os.path.dirname(self.python_executable)}:$PATH\"
        export TMPDIR=\"{self.tmp_root}\"
        export TMP=\"{self.tmp_root}\"
        export TEMP=\"{self.tmp_root}\"
        export EVOLUTION_LOG_ROOT=\"{self.evolution_log_root}\"
        export EVOLUTION_PARALLEL_SLOT=\"{self.slot_id}\"
        export EVOLUTION_DEVICE_NAME=\"{self.device_name}\"
        export PYTHONPATH=\"{self.python_override_root}:$PYTHONPATH\"
        set +u
        source \"{self.isaac_sim_setup}\"
        set -u
        cd \"{self.isaaclab_root}\"
        exec \"{self.python_executable}\" \"{self.worker_script}\" \\
          --request-dir \"{self.request_dir}\" --device \"{self.device_name}\" --headless
        """
        self._log_file = open(self.worker_log, "a", encoding="utf-8")
        self.process = subprocess.Popen(
            ["bash", "-c", shell_cmd],
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        ready_path = os.path.join(self.request_dir, "ready.json")
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if os.path.exists(ready_path):
                return
            if self.process.poll() is not None:
                raise PersistentIsaacWorkerError(f"Worker exited during startup; see {self.worker_log}")
            time.sleep(1)
        self._force_terminate()
        raise PersistentIsaacWorkerError(f"Worker did not become ready within {self.startup_timeout}s")

    def run(self, *, task_name, num_envs, run_name, checkpoint_path, max_iterations, checkpoint_interval):
        last_error = None
        for attempt in range(self.request_retries + 1):
            self._start()
            request_id = f"{int(time.time() * 1_000_000)}_{os.getpid()}"
            response_path = os.path.join(self.request_dir, f"{request_id}.response.json")
            _atomic_write_json(
                os.path.join(self.request_dir, f"{request_id}.request.json"),
                {
                    "id": request_id,
                    "task": task_name,
                    "num_envs": num_envs,
                    "device": self.device_name,
                    "run_name": run_name,
                    "checkpoint_path": checkpoint_path,
                    "max_iterations": max_iterations,
                    "checkpoint_interval": checkpoint_interval,
                },
            )
            deadline = time.time() + self.request_timeout
            last_progress = time.time()
            last_log_mtime = self._log_mtime()
            try:
                while time.time() < deadline:
                    if os.path.exists(response_path):
                        response = _load_json(response_path)
                        os.remove(response_path)
                        if not response.get("ok"):
                            raise PersistentIsaacWorkerError(
                                f"Worker failed for task={task_name}, slot={self.slot_id}: "
                                f"{response.get('error')}\n{response.get('traceback', '')}"
                            )
                        self.requests_served += 1
                        if self.max_requests and self.requests_served >= self.max_requests:
                            self.close()
                        return response
                    if self.process.poll() is not None:
                        raise PersistentIsaacWorkerError(f"Worker exited while training task={task_name}")
                    log_mtime = self._log_mtime()
                    if log_mtime > last_log_mtime:
                        last_log_mtime = log_mtime
                        last_progress = time.time()
                    if time.time() - last_progress > self.stall_timeout:
                        raise PersistentIsaacWorkerError(
                            f"Worker made no log progress for {self.stall_timeout}s while training task={task_name}"
                        )
                    time.sleep(1)
                raise PersistentIsaacWorkerError(f"Worker timed out for task={task_name}")
            except PersistentIsaacWorkerError as error:
                last_error = error
                self._remove_request_files(request_id)
                self._force_terminate()
                if attempt < self.request_retries:
                    print(f"[WORKER] Recycling slot {self.slot_id} after: {error}; retrying once.", flush=True)
        raise last_error

    def close(self):
        if self.process is not None and self.process.poll() is None:
            request_id = f"shutdown_{int(time.time() * 1_000_000)}"
            _atomic_write_json(
                os.path.join(self.request_dir, f"{request_id}.request.json"),
                {"id": request_id, "command": "shutdown"},
            )
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._force_terminate()
            else:
                self.process = None
                self._close_log_file()
        else:
            self._close_log_file()
        self.requests_served = 0
