from agents.base import ChatCallback, GooseAgent, PlannerAgent
from openai import OpenAI

from goose_game.environment import GooseEnvironment, GooseGameEnvironment, PlannerEnvironment
from goose_game.models import LevelSpec


PlannerFactory = type[PlannerAgent]
GooseFactory = type[GooseAgent]


class LevelWorkflow:
    def __init__(
        self,
        spec: LevelSpec,
        easy_mode: bool,
        client: OpenAI,
        used_model: str,
        planner_factory: PlannerFactory,
        goose_factory: GooseFactory,
        planner_append_to_chat: ChatCallback,
        goose_append_to_chat_by_id: dict[str, ChatCallback],
        max_turns: int = 20,
    ) -> None:
        self.spec = spec
        self.client = client
        self.used_model = used_model
        self.planner_factory = planner_factory
        self.goose_factory = goose_factory
        self.max_turns = max_turns
        self.planner_append_to_chat = planner_append_to_chat
        self.goose_append_to_chat_by_id = goose_append_to_chat_by_id
        self.env = GooseGameEnvironment(self.spec, easy_mode=easy_mode)
        self.env.set_max_counted_actions(self.max_turns)
        self.geese = {
            goose_id: self.goose_factory(
                self.client,
                self.used_model,
                GooseEnvironment(self.env, goose_id),
                self.goose_append_to_chat_by_id[goose_id],
            )
            for goose_id in self.spec.goose_starts
        }
        self.planner = self.planner_factory(
            self.client,
            self.used_model,
            PlannerEnvironment(self.env),
            self.geese,
            self.planner_append_to_chat,
        )

    def step(self) -> None:
        if self.env.is_complete() or not self.env.can_take_counted_action():
            return
        self.planner.step()

    def run(self) -> tuple[bool, int]:
        while not self.env.is_complete() and self.env.can_take_counted_action():
            self.step()
        return self.env.is_complete(), self.env.counted_actions
