# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SysAdmin Environment implementation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import KernelAction, KernelObservation
except ImportError:
    from models import KernelAction, KernelObservation


class MockSystem:
    """Simulates a Linux-like system state for safely running SysAdmin tasks."""

    def __init__(self):
        self.processes = {
            1: {"name": "systemd", "cpu": 0.1},
            1024: {"name": "rogue_app", "cpu": 15.5},
            2048: {"name": "sshd", "cpu": 0.2},
        }
        self.services = {
            "nginx": {"status": "inactive", "enabled": True},
            "ssh": {"status": "active", "enabled": True},
            "cron": {"status": "active", "enabled": True},
        }
        self.files = {
            "/etc/nginx/nginx.conf": "server {\n    liten 80;\n    server_name localhost;\n}",
            "/var/log/syslog": "Apr  6 09:00:00 localhost systemd[1]: Started SSH service.\n",
        }
        self.last_command_output = ""

    def run_command(self, command: str) -> tuple[str, str, int]:
        """Simple shell command parser."""
        parts = command.strip().split()
        if not parts:
            return "", "", 0

        cmd = parts[0]
        args = parts[1:]

        if cmd == "ps":
            output = "PID   COMMAND      %CPU\n"
            for pid, info in self.processes.items():
                output += f"{pid:<5} {info['name']:<12} {info['cpu']}\n"
            return output, "", 0

        elif cmd == "kill":
            if not args:
                return "", "kill: usage: kill [-s sigspec | -n signum | -sigspec] pid | jobspec ... or kill -l [sigspec]", 1
            try:
                pid = int(args[0])
                if pid in self.processes:
                    del self.processes[pid]
                    return "", "", 0
                else:
                    return "", f"kill: ({pid}) - No such process", 1
            except ValueError:
                return "", f"kill: {args[0]}: arguments must be process IDs", 1

        elif cmd == "killall":
            if not args:
                return "", "killall: usage: killall [-Z context] [-e] [-g] [-i] [-m] [-o] [-q] [-r] [-s signal] [-u user] [-v] [-w] [-I] [-V] name ...", 1
            name = args[0]
            to_kill = [pid for pid, info in self.processes.items() if info["name"] == name]
            if not to_kill:
                return "", f"{name}: no process found", 1
            for pid in to_kill:
                del self.processes[pid]
            return "", "", 0

        elif cmd == "systemctl":
            if len(args) < 2:
                return "", "systemctl: too few arguments", 1
            action = args[0]
            service = args[1]
            if service not in self.services:
                return "", f"Failed to {action} {service}.service: Unit {service}.service not found.", 5

            if action == "status":
                status = self.services[service]["status"]
                output = f"● {service}.service\n   Loaded: loaded\n   Active: {status}\n"
                return output, "", 0
            elif action == "start":
                # Special check for nginx: if config is broken, it fails to start
                if service == "nginx" and "liten" in self.files.get("/etc/nginx/nginx.conf", ""):
                    return "", "Job for nginx.service failed because the control process exited with error-code.", 1
                self.services[service]["status"] = "active"
                return "", "", 0
            elif action == "stop":
                self.services[service]["status"] = "inactive"
                return "", "", 0
            elif action == "restart":
                if service == "nginx" and "liten" in self.files.get("/etc/nginx/nginx.conf", ""):
                    return "", "Job for nginx.service failed. See 'systemctl status nginx.service' and 'journalctl -xe' for details.", 1
                self.services[service]["status"] = "active"
                return "", "", 0

        elif cmd == "cat":
            if not args:
                return "", "cat: usage: cat [file ...]", 1
            path = args[0]
            if path in self.files:
                return self.files[path], "", 0
            return "", f"cat: {path}: No such file or directory", 1

        elif cmd == "sed":
            # Very simple implementation for 'sed -i 's/old/new/g' file'
            if "-i" in args:
                try:
                    pattern_idx = args.index("-i") + 1
                    file_idx = pattern_idx + 1
                    pattern = args[pattern_idx]
                    path = args[file_idx]
                    
                    if path not in self.files:
                        return "", f"sed: can't read {path}: No such file or directory", 2
                        
                    match = re.match(r"s/(.*)/(.*)/g", pattern.strip("'"))
                    if match:
                        old, new = match.groups()
                        self.files[path] = self.files[path].replace(old, new)
                        return "", "", 0
                except (ValueError, IndexError):
                    pass
            return "", "sed: invalid option or pattern", 1

        return "", f"bash: {cmd}: command not found", 127


class KernelEnvironment(Environment):
    """
    Real-world SysAdmin troubleshooting environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        *,
        max_steps: int = 15,
        transform: Optional[Any] = None,
        rubric: Optional[Any] = None,
    ):
        super().__init__(transform=transform, rubric=rubric)
        self._max_steps = max_steps
        self._reset_count = 0
        self._terminated = False
        self._cumulative_reward = 0.0
        self._system = MockSystem()
        self._state = self._build_state(episode_id=str(uuid4()))

    def _build_state(self, *, episode_id: str, step_count: int = 0) -> State:
        return State(
            episode_id=episode_id,
            step_count=step_count,
            terminated=self._terminated,
            max_steps=self._max_steps,
            cumulative_reward=round(self._cumulative_reward, 4),
            reset_count=self._reset_count,
        )

    def _get_system_summary(self) -> Dict[str, Any]:
        return {
            "active_services": [s for s, v in self._system.services.items() if v["status"] == "active"],
            "running_processes": [p["name"] for p in self._system.processes.values()],
            "nginx_config_ok": "liten" not in self._system.files.get("/etc/nginx/nginx.conf", ""),
        }

    def _get_tasks_status(self) -> Dict[str, bool]:
        summary = self._get_system_summary()
        return {
            "task_1_kill_rogue": "rogue_app" not in summary["running_processes"],
            "task_2_nginx_active": "nginx" in summary["active_services"],
            "task_3_nginx_config_fixed": summary["nginx_config_ok"],
        }

    def _compute_reward(self) -> float:
        status = self._get_tasks_status()
        # Weights: 0.2, 0.3, 0.5
        reward = 0.0
        if status["task_1_kill_rogue"]:
            reward += 0.2
        if status["task_2_nginx_active"]:
            reward += 0.3
        if status["task_3_nginx_config_fixed"]:
            reward += 0.5
            
        # We return the total progress as the reward (normalized 0-1)
        # However, OpenEnv usually expects per-step reward or cumulative.
        # Let's make it cumulative-friendly.
        current_progress = reward
        step_reward = max(0.0, current_progress - self._cumulative_reward)
        return round(step_reward, 4)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **_: Any,
    ) -> KernelObservation:
        del seed
        self._reset_rubric()
        self._reset_count += 1
        self._terminated = False
        self._cumulative_reward = 0.0
        self._system = MockSystem()
        self._state = self._build_state(episode_id=episode_id or str(uuid4()))

        return KernelObservation(
            stdout="System boot complete. Welcome to SysAdmin Shell.\nTasks:\n1. [Easy] Kill the rogue_app process.\n2. [Medium] Start the nginx service.\n3. [Hard] Fix the typo in /etc/nginx/nginx.conf ('liten' -> 'listen') and restart nginx.",
            system_state=self._get_system_summary(),
            tasks_status=self._get_tasks_status(),
            done=False,
            reward=0.0,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
            },
        )

    def step(
        self,
        action: KernelAction,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> KernelObservation:
        if self._terminated:
            raise RuntimeError("episode is terminated; call reset() before step() again")

        self._state.step_count += 1
        
        stdout, stderr, exit_code = self._system.run_command(action.command)
        
        step_reward = self._compute_reward()
        self._cumulative_reward += step_reward
        
        tasks = self._get_tasks_status()
        self._terminated = (self._state.step_count >= self._max_steps) or all(tasks.values())

        observation = KernelObservation(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            system_state=self._get_system_summary(),
            tasks_status=tasks,
            done=self._terminated,
            reward=step_reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "cumulative_reward": round(self._cumulative_reward, 4),
            },
        )
        
        self._state = self._build_state(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
        )

        return self._apply_transform(observation)

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.description = "A real-world SysAdmin troubleshooting environment where an agent identifies and fixes system issues using shell commands."
        return metadata
