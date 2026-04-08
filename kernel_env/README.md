---
title: Sysadmin Troubleshooter
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# SysAdmin Troubleshooting Environment

A real-world OpenEnv environment where an RL agent acts as a Junior System Administrator. The agent interacts with a mock Linux shell to identify and resolve common server issues.

## Environment Details

### Action Space
**SysAdminAction**: A single shell command string.
- `command` (str): The shell command to execute (e.g., `ps aux`, `systemctl start nginx`, `kill 1024`).

### Observation Space
**SysAdminObservation**: Returns the result of the command and the current system state.
- `stdout` (str): Standard output from the command.
- `stderr` (str): Standard error from the command.
- `exit_code` (int): Return code of the command.
- `system_state` (dict): Summary of active services and running processes.
- `tasks_status` (dict): Boolean status of the three target tasks.
- `reward` (float): Partial progress reward (0.0 to 1.0 cumulative).
- `done` (bool): True if all tasks are complete or max steps reached.

## Tasks

The environment includes three tasks of increasing difficulty:

1.  **[Easy] Rogue Process Cleanup**: Identify a high-CPU process (`rogue_app`) using `ps` and terminate it using `kill` or `killall`.
2.  **[Medium] Service Recovery**: The `nginx` service is currently inactive. The agent must identify this and start the service using `systemctl start nginx`.
3.  **[Hard] Configuration Fix**: The `nginx` configuration has a typo (`liten` instead of `listen`). The agent must read the config file (`cat /etc/nginx/nginx.conf`), fix the typo (e.g., using `sed`), and restart the service.

## Reward Function

The reward is based on task completion:
- **Task 1**: +0.2
- **Task 2**: +0.3
- **Task 3**: +0.5
Total potential reward: **1.0**.

## Quick Start

### 1. Build and Start the Environment

```bash
# Build the Docker image
docker build -t sysadmin-env:latest -f server/Dockerfile .

# Run the container
docker run -p 8000:8000 sysadmin-env:latest
```

### 2. Run Inference

Ensure you have the required environment variables set:

```bash
export API_BASE_URL="your-api-endpoint"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"

python inference.py
```

## Spec Compliance

This environment implements the full OpenEnv spec:
- Typed Pydantic models for Actions and Observations.
- Standard `step()`, `reset()`, and `state()` endpoints.
- Valid `openenv.yaml` manifest.
- reproducible `inference.py` with mandatory `[START]`, `[STEP]`, and `[END]` logging.
