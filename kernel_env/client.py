# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import KernelAction, KernelObservation
except ImportError:
    try:
        from .models import KernelAction, KernelObservation
    except ImportError:
        from kernel_env.models import KernelAction, KernelObservation



class KernelEnv(
    EnvClient[KernelAction, KernelObservation, State]
):
    """
    Client for the Kernel Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with KernelEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(KernelAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = KernelEnv.from_docker_image("kernel_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(KernelAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: KernelAction) -> Dict:
        """
        Convert KernelAction to JSON payload for step message.
        """
        return {
            "command": action.command,
        }

    def _parse_result(self, payload: Dict) -> StepResult[KernelObservation]:
        """
        Parse server response into StepResult[KernelObservation].
        """
        obs_data = payload.get("observation", {})
        observation = KernelObservation(
            stdout=obs_data.get("stdout", ""),
            stderr=obs_data.get("stderr", ""),
            exit_code=obs_data.get("exit_code", 0),
            system_state=obs_data.get("system_state", {}),
            tasks_status=obs_data.get("tasks_status", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        state_payload = dict(payload)
        episode_id = state_payload.pop("episode_id", None)
        step_count = state_payload.pop("step_count", 0)
        return State(
            episode_id=episode_id,
            step_count=step_count,
            **state_payload,
        )
