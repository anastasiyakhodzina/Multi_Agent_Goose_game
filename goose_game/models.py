from dataclasses import dataclass, field
from enum import Enum


Position = tuple[int, int]


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(slots=True)
class Door:
    door_id: str
    position: Position
    is_open: bool = False
    color: str | None = None


@dataclass(slots=True)
class Button:
    button_id: str
    position: Position
    label: str
    targets: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HonkEvent:
    goose_id: str
    position: Position
    count: int = 1


@dataclass(slots=True)
class LevelSpec:
    level_id: int
    name: str
    task_description: str
    width: int
    height: int
    walkable_cells: set[Position]
    goose_starts: dict[str, Position]
    goal_cells: set[Position]
    doors: dict[str, Door] = field(default_factory=dict)
    buttons: dict[str, Button] = field(default_factory=dict)
    special_cells: dict[str, list[Position]] = field(default_factory=dict)
