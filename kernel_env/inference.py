"""
inference.py — ThinkForge SysAdmin Agent
Meta PyTorch Hackathon x Scaler School of Technology, Round 1

Runs an RL episode: connects to the KernelEnv server, resets, and
drives the LLM agent through up to 15 steps of shell commands.

All network and parsing errors are caught so the script always exits 0.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Import KernelEnv client / models — handle all possible install layouts
# ---------------------------------------------------------------------------
try:
    from kernel_env.client import KernelEnv
    from kernel_env.models import KernelAction
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    try:
        from client import KernelEnv
        from models import KernelAction
    except ImportError:
        sys.path.insert(0, os.path.join(_here, "kernel_env"))
        from client import KernelEnv
        from models import KernelAction

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:8000")

# Max steps per episode
MAX_STEPS = 15
# Connection retry settings
MAX_CONNECT_RETRIES = 5
CONNECT_RETRY_DELAY = 3.0  # seconds between retries


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
def get_llm_response(client: OpenAI, prompt: str) -> str:
    """Call the LLM and return only the shell command text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Linux System Administrator. "
                        "Your task is to solve issues in a mock server environment. "
                        "Respond with ONLY the single shell command to execute. "
                        "No explanations, no markdown fences, no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        cmd = response.choices[0].message.content.strip()
        # Strip accidental markdown code fences
        for fence in ["```bash\n", "```sh\n", "```\n", "```"]:
            cmd = cmd.replace(fence, "")
        return cmd.strip()
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}. Using fallback 'ps aux'.")
        return "ps aux"


# ---------------------------------------------------------------------------
# Connection helper with retry
# ---------------------------------------------------------------------------
async def connect_with_retry(env_url: str) -> Any:
    """
    Attempt to connect to the environment server with retries.
    Returns a connected KernelEnv async context, or raises on final failure.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_CONNECT_RETRIES + 1):
        try:
            print(f"[CONNECT] Attempt {attempt}/{MAX_CONNECT_RETRIES} → {env_url}")
            env = KernelEnv(base_url=env_url, connect_timeout_s=15.0, message_timeout_s=90.0)
            await env.connect()
            print(f"[CONNECT] Connected successfully on attempt {attempt}.")
            return env
        except Exception as exc:
            last_exc = exc
            print(f"[CONNECT] Attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt < MAX_CONNECT_RETRIES:
                await asyncio.sleep(CONNECT_RETRY_DELAY)
    raise ConnectionError(
        f"Could not connect to {env_url} after {MAX_CONNECT_RETRIES} attempts. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
async def run_inference() -> None:
    """Full RL episode: reset → loop(step) → log results."""

    # Warn about missing env vars but do not abort — validator may inject them
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(v)]
    if missing:
        print(f"[WARN] Missing environment variables: {missing}. Continuing anyway.")

    # Build OpenAI client (safe even if creds are empty — will fail gracefully at call time)
    llm_client = OpenAI(
        base_url=API_BASE_URL or None,
        api_key=HF_TOKEN or "placeholder",
    )

    env = None
    try:
        # ── 1. Connect ──────────────────────────────────────────────────────
        env = await connect_with_retry(ENV_URL)

        # ── 2. Reset ────────────────────────────────────────────────────────
        print("[RESET] Resetting environment...")
        reset_result = await env.reset()

        # reset() returns StepResult; .observation is a KernelObservation
        observation = reset_result.observation

        metadata: dict = getattr(observation, "metadata", {}) or {}
        episode_id: str = metadata.get("episode_id", str(uuid.uuid4()))

        start_log = {
            "episode_id": episode_id,
            "timestamp": time.time(),
            "tasks": getattr(observation, "tasks_status", {}),
        }
        print(f"[START] {json.dumps(start_log)}")

        # ── 3. Episode loop ─────────────────────────────────────────────────
        done: bool = bool(getattr(observation, "done", False))
        step_count: int = 0
        total_reward: float = 0.0

        while not done and step_count < MAX_STEPS:
            step_count += 1

            tasks_status = getattr(observation, "tasks_status", {})
            system_state = getattr(observation, "system_state", {})
            stdout = getattr(observation, "stdout", "")
            stderr = getattr(observation, "stderr", "")
            exit_code = getattr(observation, "exit_code", 0)

            prompt = (
                f"Current Tasks: {json.dumps(tasks_status)}\n"
                f"System State: {json.dumps(system_state)}\n"
                f"Last stdout:\n{stdout}\n"
                f"Last stderr:\n{stderr}\n"
                f"Exit code: {exit_code}\n\n"
                "What is your next shell command to fix the remaining tasks?"
            )

            # Get command from LLM
            command = get_llm_response(llm_client, prompt)
            print(f"[ACTION] Step {step_count}: {command!r}")

            # Execute step
            step_result = await env.step(KernelAction(command=command))

            observation = step_result.observation
            reward: float = float(getattr(step_result, "reward", 0.0))
            done = bool(getattr(step_result, "done", False))
            total_reward += reward

            step_log = {
                "step": step_count,
                "command": command,
                "stdout": getattr(observation, "stdout", ""),
                "stderr": getattr(observation, "stderr", ""),
                "exit_code": getattr(observation, "exit_code", 0),
                "reward": round(reward, 4),
                "done": done,
                "tasks_status": getattr(observation, "tasks_status", {}),
            }
            print(f"[STEP] {json.dumps(step_log)}")

        # ── 4. End log ──────────────────────────────────────────────────────
        tasks_status_final = getattr(observation, "tasks_status", {})
        success = bool(tasks_status_final) and all(tasks_status_final.values())

        end_log = {
            "episode_id": episode_id,
            "total_reward": round(total_reward, 4),
            "steps": step_count,
            "success": success,
        }
        print(f"[END] {json.dumps(end_log)}")

    except ConnectionError as exc:
        # Environment server unreachable — log and exit cleanly (exit code 0)
        print(f"[ERROR] Connection failed: {exc}")
        print("[INFO] Exiting cleanly — environment server was not reachable.")

    except Exception as exc:
        # Catch-all: log the full traceback for debugging but do NOT re-raise
        import traceback
        print(f"[ERROR] Unhandled exception during inference: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("[INFO] Exiting cleanly after error.")

    finally:
        # Always close the connection
        if env is not None:
            try:
                await env.close()
                print("[CLOSE] Environment connection closed.")
            except Exception as close_exc:
                print(f"[WARN] Error while closing environment: {close_exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run_inference())
    # Script always exits with code 0
