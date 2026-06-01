from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

from goose_game.environment import GooseEnvironment, PlannerEnvironment

ChatCallback = Callable[[str], None]


@dataclass(slots=True)
class GooseAgentMessage:
    description: str


@dataclass(slots=True)
class GooseAgentResult:
    output: str
    error: str | None = None


class GooseAgent(ABC):
    def __init__(self, client: OpenAI, used_model: str, env: GooseEnvironment, append_to_chat: ChatCallback) -> None:
        self._client = client
        self._used_model = used_model
        self._env = env
        self._append_to_chat = append_to_chat

    @abstractmethod
    def on_call(self, message: GooseAgentMessage) -> GooseAgentResult:
        pass


class PlannerAgent(ABC):
    def __init__(
        self,
        client: OpenAI,
        used_model: str,
        env: PlannerEnvironment,
        agents: dict[str, GooseAgent],
        append_to_chat: ChatCallback,
    ) -> None:
        self._client = client
        self._used_model = used_model
        self._env = env
        self._agents = dict(agents)
        self._append_to_chat = append_to_chat

    @abstractmethod
    def step(self) -> None:
        pass
