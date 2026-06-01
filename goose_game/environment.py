from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from goose_game.models import Button, Direction, Door, HonkEvent, LevelSpec, Position
from goose_game.render import render_ascii_map


ActionCallback = Callable[[], None]


def move_position(position: Position, direction: Direction) -> Position:
    row, col = position
    if direction == Direction.UP:
        return (row - 1, col)
    if direction == Direction.DOWN:
        return (row + 1, col)
    if direction == Direction.LEFT:
        return (row, col - 1)
    return (row, col + 1)


@dataclass(slots=True)
class GameState:
    goose_positions: dict[str, Position]
    doors: dict[str, Door]
    buttons: dict[str, Button]
    final_honked_geese: set[str] = field(default_factory=set)


class GooseGameEnvironment:
    def __init__(self, spec: LevelSpec, easy_mode: bool = True) -> None:
        self.spec = spec
        self.easy_mode = easy_mode
        self.state = GameState(
            goose_positions=dict(spec.goose_starts),
            doors={
                door_id: Door(door_id=door.door_id, position=door.position, is_open=door.is_open, color=door.color)
                for door_id, door in spec.doors.items()
            },
            buttons={
                button_id: Button(
                    button_id=button.button_id,
                    position=button.position,
                    label=button.label,
                    targets=list(button.targets),
                )
                for button_id, button in spec.buttons.items()
            },
        )
        self._honk_events: list[HonkEvent] = []
        self._action_callbacks: list[ActionCallback] = []
        self._max_counted_actions: int | None = None
        self._counted_actions = 0
        self._sync_doors_with_buttons()

    def register_action_callback(self, callback: ActionCallback) -> None:
        self._action_callbacks.append(callback)

    def set_max_counted_actions(self, max_counted_actions: int) -> None:
        self._max_counted_actions = max_counted_actions

    @property
    def counted_actions(self) -> int:
        return self._counted_actions

    def can_take_counted_action(self) -> bool:
        return self._max_counted_actions is None or self._counted_actions < self._max_counted_actions

    def _consume_counted_action(self) -> bool:
        if not self.can_take_counted_action():
            return False
        self._counted_actions += 1
        return True

    def _notify_action_callbacks(self) -> None:
        for callback in self._action_callbacks:
            callback()

    def in_bounds(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.spec.height and 0 <= col < self.spec.width

    def is_walkable(self, position: Position) -> bool:
        if position not in self.spec.walkable_cells:
            return False
        for door in self.state.doors.values():
            if door.position == position and not door.is_open:
                return False
        return True

    def move_goose(self, goose_id: str, direction: Direction) -> bool:
        next_position = move_position(self.state.goose_positions[goose_id], direction)
        if not self.in_bounds(next_position) or not self.is_walkable(next_position):
            return False
        if not self._consume_counted_action():
            return False
        self.state.goose_positions[goose_id] = next_position
        self._sync_doors_with_buttons()
        self._notify_action_callbacks()
        return True

    def honk(self, goose_id: str, count: int) -> HonkEvent:
        if not self._consume_counted_action():
            raise RuntimeError("Action limit reached.")
        event = HonkEvent(goose_id=goose_id, position=self.state.goose_positions[goose_id], count=count)
        self._honk_events.append(event)
        if self.state.goose_positions[goose_id] in self.spec.goal_cells:
            self.state.final_honked_geese.add(goose_id)
        self._notify_action_callbacks()
        return event

    def _pressed_buttons(self) -> list[Button]:
        goose_positions = set(self.state.goose_positions.values())
        return [button for button in self.state.buttons.values() if button.position in goose_positions]

    def _sync_doors_with_buttons(self) -> None:
        for door_id, door in self.state.doors.items():
            default_door = self.spec.doors[door_id]
            door.is_open = default_door.is_open

        for button in self._pressed_buttons():
            for door_id in button.targets:
                door = self.state.doors.get(door_id)
                if door is None:
                    continue
                door.is_open = True

    def _door_by_position(self) -> dict[Position, Door]:
        return {door.position: door for door in self.state.doors.values()}

    def _visible_positions_for_goose(self, goose_id: str) -> set[Position]:
        start = self.state.goose_positions[goose_id]
        visible_positions: set[Position] = {start}
        reachable_positions: set[Position] = {start}
        queue: deque[Position] = deque([start])
        door_by_position = self._door_by_position()

        while queue:
            position = queue.popleft()
            row, col = position
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if not self.in_bounds(neighbor):
                    continue
                visible_positions.add(neighbor)
                door = door_by_position.get(neighbor)
                if door is not None:
                    if door.is_open and neighbor not in reachable_positions:
                        reachable_positions.add(neighbor)
                        queue.append(neighbor)
                    continue
                if neighbor in self.spec.walkable_cells:
                    if neighbor not in reachable_positions:
                        reachable_positions.add(neighbor)
                        queue.append(neighbor)
                    visible_positions.add(neighbor)

        return visible_positions

    def visible_positions_for_goose(self, goose_id: str) -> set[Position]:
        if self.easy_mode:
            return {(row, col) for row in range(self.spec.height) for col in range(self.spec.width)}
        return self._visible_positions_for_goose(goose_id)

    def visible_positions_for_display(self) -> set[Position]:
        if self.easy_mode:
            return {(row, col) for row in range(self.spec.height) for col in range(self.spec.width)}

        visible_positions: set[Position] = set()
        for goose_id in self.state.goose_positions:
            visible_positions.update(self._visible_positions_for_goose(goose_id))
        return visible_positions

    def describe_goose_state(self, goose_id: str) -> str:
        return render_ascii_map(
            self.spec,
            self.state.goose_positions,
            self.state.buttons.values(),
            self.state.doors.values(),
            visible_positions=self.visible_positions_for_goose(goose_id),
        )

    def is_complete(self) -> bool:
        if not all(position in self.spec.goal_cells for position in self.state.goose_positions.values()):
            return False
        return set(self.state.goose_positions) <= self.state.final_honked_geese


class PlannerEnvironment:
    def __init__(self, env: GooseGameEnvironment) -> None:
        # WARNING:
        # These fields are superprivate - you are not allowed to use them in your solution!
        # Solutions that use them will get 0 points.
        self.__env = env

    @property
    def level_name(self) -> str:
        return self.__env.spec.name

    @property
    def task_description(self) -> str:
        return self.__env.spec.task_description


class GooseEnvironment:
    def __init__(self, env: GooseGameEnvironment, goose_id: str) -> None:
        # WARNING:
        # These fields are superprivate - you are not allowed to use them in your solution!
        # Solutions that use them will get 0 points.
        self.__env = env
        self.__goose_id = goose_id

    @property
    def goose_id(self) -> str:
        return self.__goose_id

    def can_take_counted_action(self) -> bool:
        return self.__env.can_take_counted_action()

    def visible_goose_positions(self) -> dict[str, Position]:
        visible_positions = self.__env.visible_positions_for_goose(self.__goose_id)
        return {
            goose_id: position
            for goose_id, position in self.__env.state.goose_positions.items()
            if position in visible_positions
        }

    def describe_state(self) -> str:
        return self.__env.describe_goose_state(self.__goose_id)

    def move(self, direction: Direction) -> bool:
        return self.__env.move_goose(self.__goose_id, direction)

    def honk(self, count: int) -> HonkEvent:
        return self.__env.honk(self.__goose_id, count=count)
