import os
import json
import uuid
import time
import asyncio
import inspect
from typing import Any, Dict, List
from openai import OpenAI

try:
    from kernel_env.client import KernelEnv
    from kernel_env.models import KernelAction
except ImportError:
    # Handle flattened structure (HF) or different paths
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    try:
        from client import KernelEnv
        from models import KernelAction
    except ImportError:
        # Last resort for standard execution
        sys.path.append(os.path.join(current_dir, "kernel_env"))
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

async def get_observation(result_or_obs: Any) -> Any:
    """
    Bulletproof helper to extract the observation from a StepResult or raw object,
    handling any unexpected coroutines.
    """
    # 1. If the result itself is a coroutine (unlikely after await, but safe), await it
    if inspect.iscoroutine(result_or_obs):
        result_or_obs = await result_or_obs
    
    # 2. Extract observation from StepResult if present, otherwise assume it's the observation
    observation = getattr(result_or_obs, "observation", result_or_obs)
    
    # 3. If the observation field itself is a coroutine, await it
    if inspect.iscoroutine(observation):
        observation = await observation
        
    return observation

async def run_inference():
    # Initialize OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    # Initialize Environment
    env_url = os.environ.get("ENV_URL", "http://localhost:8000")
    
    async with KernelEnv(base_url=env_url) as env:
        try:
            # Reset Environment
            print(f"Connecting to environment at {env_url}...")
            raw_result = await env.reset()
            observation = await get_observation(raw_result)
            
            # Extract metadata safely
            metadata = getattr(observation, "metadata", {})
            episode_id = metadata.get("episode_id", str(uuid.uuid4()))
            
            # [START] Logging
            start_log = {
                "episode_id": episode_id,
                "timestamp": time.time(),
                "tasks": getattr(observation, "tasks_status", {}),
            }
            print(f"[START] {json.dumps(start_log)}")

            done = False
            step_count = 0
            total_reward = 0.0

            while not done and step_count < 15:
                step_count += 1
                
                # Construct Prompt
                tasks_status = getattr(observation, "tasks_status", {})
                system_state = getattr(observation, "system_state", {})
                stdout = getattr(observation, "stdout", "")
                stderr = getattr(observation, "stderr", "")
                exit_code = getattr(observation, "exit_code", 0)

                prompt = (
                    f"Current Tasks: {tasks_status}\n"
                    f"System State: {system_state}\n"
                    f"Last stdout: {stdout}\n"
                    f"Last stderr: {stderr}\n"
                    f"Exit code: {exit_code}\n\n"
                    "What is your next command?"
                )
                
                # Get Action from LLM
                command = get_llm_response(client, prompt)
                
                # Execute Step
                raw_step_result = await env.step(KernelAction(command=command))
                observation = await get_observation(raw_step_result)
                
                # Extract reward and done status based on result type
                reward = getattr(raw_step_result, "reward", getattr(observation, "reward", 0.0))
                if inspect.iscoroutine(reward):
                    reward = await reward
                    
                done = getattr(raw_step_result, "done", getattr(observation, "done", False))
                if inspect.iscoroutine(done):
                    done = await done
                
                total_reward += float(reward)
                
                # [STEP] Logging
                step_log = {
                    "step": step_count,
                    "command": command,
                    "stdout": getattr(observation, "stdout", ""),
                    "stderr": getattr(observation, "stderr", ""),
                    "exit_code": getattr(observation, "exit_code", 0),
                    "reward": float(reward),
                    "done": bool(done),
                    "tasks_status": getattr(observation, "tasks_status", {}),
                }
                print(f"[STEP] {json.dumps(step_log)}")
                
            # [END] Logging
            end_log = {
                "episode_id": episode_id,
                "total_reward": round(total_reward, 4),
                "steps": step_count,
                "success": all(getattr(observation, "tasks_status", {}).values()) if hasattr(observation, "tasks_status") else False,
            }
            print(f"[END] {json.dumps(end_log)}")

        except Exception as e:
            print(f"Error during inference: {type(e).__name__}: {e}")
            # print traceback if needed for validator logs
            import traceback
            traceback.print_exc()
            raise e

if __name__ == "__main__":
    if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
        print("Warning: Missing required environment variables (API_BASE_URL, MODEL_NAME, HF_TOKEN)")
    
    asyncio.run(run_inference())
