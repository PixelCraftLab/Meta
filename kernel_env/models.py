# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Kernel Env environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


class KernelAction(Action):
    """Action payload for a single environment step (a shell command)."""

    command: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="The shell command to execute (e.g., 'ps aux', 'kill 1234').",
    )

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        """Reject blank commands while preserving the original content."""
        if not value.strip():
            raise ValueError("command must contain at least one non-whitespace character")
        return value


class KernelObservation(Observation):
    """Observation returned by the SysAdmin environment."""

    stdout: str = Field(default="", description="Standard output from the executed command")
    stderr: str = Field(default="", description="Standard error from the executed command")
    exit_code: int = Field(default=0, description="Exit code of the executed command")
    system_state: dict = Field(
        default_factory=dict,
        description="Summary of current system state (CPU, Mem, Services)",
    )
    tasks_status: dict = Field(
        default_factory=dict,
        description="Completion status of required tasks",
    )
