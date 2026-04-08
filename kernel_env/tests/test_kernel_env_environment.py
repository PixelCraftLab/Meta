from __future__ import annotations

import pytest

from models import KernelAction
from server.kernel_env_environment import KernelEnvironment


def test_reset_initializes_episode_state() -> None:
    env = KernelEnvironment(max_steps=3, target_message_length=12)

    observation = env.reset(episode_id="episode-1")

    assert observation.echoed_message == "Kernel Env environment ready."
    assert observation.reward == 0.0
    assert observation.done is False
    assert observation.metadata["episode_id"] == "episode-1"
    assert env.state.episode_id == "episode-1"
    assert env.state.step_count == 0
    assert env.state.max_steps == 3
    assert env.state.cumulative_reward == 0.0


def test_step_updates_state_and_returns_reward_metadata() -> None:
    env = KernelEnvironment(max_steps=3, target_message_length=12)
    env.reset()

    observation = env.step(KernelAction(message="hello   openenv"))

    assert observation.echoed_message == "hello   openenv"
    assert observation.normalized_message == "hello openenv"
    assert observation.message_length == len("hello openenv")
    assert observation.reward is not None
    assert observation.reward > 0
    assert observation.metadata["step_count"] == 1
    assert observation.metadata["remaining_steps"] == 2
    assert env.state.step_count == 1
    assert env.state.last_message == "hello openenv"


def test_repeating_message_receives_lower_reward() -> None:
    env = KernelEnvironment(max_steps=3, target_message_length=12)
    env.reset()

    first = env.step(KernelAction(message="balanced text"))
    second = env.step(KernelAction(message="balanced text"))

    assert first.reward is not None
    assert second.reward is not None
    assert second.reward < first.reward


def test_environment_terminates_at_max_steps() -> None:
    env = KernelEnvironment(max_steps=2, target_message_length=12)
    env.reset()

    first = env.step(KernelAction(message="first move"))
    second = env.step(KernelAction(message="second move"))

    assert first.done is False
    assert second.done is True
    assert second.metadata["termination_reason"] == "max_steps_reached"
    assert env.state.terminated is True

    with pytest.raises(RuntimeError, match="call reset"):
        env.step(KernelAction(message="third move"))


def test_blank_messages_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        KernelAction(message="   ")
