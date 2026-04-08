import os
import json
import uuid
import time
from typing import Any, Dict, List
from openai import OpenAI

try:
    from kernel_env.client import KernelEnv
    from kernel_env.models import KernelAction
except ImportError:
    # If running from within the project root
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from client import KernelEnv
    from models import KernelAction

# Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")

def get_llm_response(client: OpenAI, prompt: str) -> str:
    """Gets a command from the LLM based on the current state."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert Linux System Administrator. Your task is to solve issues in a mock server. "
                                         "Provide ONLY the shell command to execute. No explanations or extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

def run_inference():
    # Initialize OpenAI client with HF Token (if applicable)
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    # Initialize Environment
    # We assume the environment is running locally or at a provided URL
    # For the hackathon evaluation, it will likely be at localhost:8000
    env_url = os.environ.get("ENV_URL", "http://localhost:8000")
    env = KernelEnv(base_url=env_url)

    try:
        # Reset Environment
        observation = env.reset()
        episode_id = observation.metadata.get("episode_id", str(uuid.uuid4()))
        
        # [START] Logging
        start_log = {
            "episode_id": episode_id,
            "timestamp": time.time(),
            "tasks": observation.tasks_status,
        }
        print(f"[START] {json.dumps(start_log)}")

        done = False
        step_count = 0
        total_reward = 0.0

        while not done and step_count < 15:
            step_count += 1
            
            # Construct Prompt
            prompt = (
                f"Current Tasks: {observation.tasks_status}\n"
                f"System State: {observation.system_state}\n"
                f"Last stdout: {observation.stdout}\n"
                f"Last stderr: {observation.stderr}\n"
                f"Exit code: {observation.exit_code}\n\n"
                "What is your next command?"
            )
            
            # Get Action from LLM
            command = get_llm_response(client, prompt)
            
            # Execute Step
            observation = env.step(KernelAction(command=command))
            reward = observation.reward
            total_reward += reward
            done = observation.done
            
            # [STEP] Logging
            step_log = {
                "step": step_count,
                "command": command,
                "stdout": observation.stdout,
                "stderr": observation.stderr,
                "exit_code": observation.exit_code,
                "reward": reward,
                "done": done,
                "tasks_status": observation.tasks_status,
            }
            print(f"[STEP] {json.dumps(step_log)}")
            
        # [END] Logging
        end_log = {
            "episode_id": episode_id,
            "total_reward": round(total_reward, 4),
            "steps": step_count,
            "success": all(observation.tasks_status.values()),
        }
        print(f"[END] {json.dumps(end_log)}")

    finally:
        env.close()

if __name__ == "__main__":
    if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
        print("Missing required environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN")
        # For local testing, we might want to exit or provide defaults
        # sys.exit(1)
    run_inference()
