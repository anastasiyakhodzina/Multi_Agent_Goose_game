import random

from goose_game.maps import parse_ascii_map
from goose_game.models import Button, Door, LevelSpec

LEVEL_RANDOM_SEED = 0


def _sorted_positions(level: LevelSpec, symbol: str) -> list[tuple[int, int]]:
    return sorted(level.special_cells.get(symbol, []))


def _positions_in_row_range(positions: list[tuple[int, int]], row_start: int, row_end: int) -> list[tuple[int, int]]:
    return [position for position in positions if row_start <= position[0] <= row_end]


def _paired_positions_by_column(positions: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    grouped: dict[int, list[tuple[int, int]]] = {}
    for position in positions:
        grouped.setdefault(position[1], []).append(position)
    return [sorted(grouped[col]) for col in sorted(grouped)]


def level_from_ascii(level_id: int, name: str, task_description: str, map_text: str) -> LevelSpec:
    parsed = parse_ascii_map(map_text)
    return LevelSpec(
        level_id=level_id,
        name=name,
        task_description=task_description,
        width=parsed.width,
        height=parsed.height,
        walkable_cells=parsed.walkable_cells,
        goose_starts=parsed.goose_starts,
        goal_cells=set(parsed.special_cells["*"]),
        special_cells=parsed.special_cells,
    )


def build_level_1() -> LevelSpec:
    return level_from_ascii(
        1,
        "Shared Map Final Honk",
        "The symbol * is the goal cell. Both geese must reach * and both honk there.",
        """
X....
.....
....*
.....
Y....
""",
    )


def build_level_2() -> LevelSpec:
    level = level_from_ascii(
        2,
        "Button Rescue",
        "The geese start in separate regions. The symbol * is the goal cell. Goose 1 must stand on the button to keep the door open for Goose 2, then both geese must reach * and honk there.",
        """
...#....
.X.#.Y..
@..#....
...##$##
...*....
""",
    )
    button_position = _sorted_positions(level, "@")[0]
    door_position = _sorted_positions(level, "$")[0]
    level.buttons["button_free"] = Button("button_free", position=button_position, label="Free Door", targets=["door_free"])
    level.doors["door_free"] = Door("door_free", position=door_position, is_open=False)
    return level


def build_level_3() -> LevelSpec:
    rng = random.Random(LEVEL_RANDOM_SEED)
    level = level_from_ascii(
        3,
        "Single Hidden Button",
        "The symbol * is the goal cell. Exactly one of the three buttons opens the door to the goal while occupied, so the agents must test the buttons and coordinate.",
        """
.*..#....
@...#....
....$..Y.
@...#....
....#####
@....X...
""",
    )
    button_positions = _sorted_positions(level, "@")
    winning_button_index = rng.randrange(len(button_positions))
    door_position = _sorted_positions(level, "$")[0]
    level.doors["goal_door"] = Door("goal_door", position=door_position, is_open=False)
    for index, position in enumerate(button_positions, start=1):
        targets = ["goal_door"] if index - 1 == winning_button_index else []
        level.buttons[f"button_{index}"] = Button(f"button_{index}", position=position, label=f"button_{index}", targets=targets)
    return level


def build_level_4() -> LevelSpec:
    rng = random.Random(LEVEL_RANDOM_SEED)
    level = level_from_ascii(
        4,
        "Unknown Door Wiring",
        "The symbol * is the goal cell. Goose 1 must pass through three doors in order to reach *. Goose 2 has three pressure buttons with hidden door connections, so they must experiment and communicate through the planner.",
        """
###########
#Y@.@.@..*#
#.........#
###.#######
#..$.$.$X.#
###########
""",
    )
    door_ids = ["door_a", "door_b", "door_c"]
    button_ids = ["button_1", "button_2", "button_3"]
    button_targets = door_ids[:]
    rng.shuffle(button_targets)
    for index, position in enumerate(_sorted_positions(level, "$")[:3]):
        level.doors[door_ids[index]] = Door(door_ids[index], position=position, is_open=False)
    for index, position in enumerate(_sorted_positions(level, "@")[:3]):
        level.buttons[button_ids[index]] = Button(
            button_ids[index],
            position=position,
            label=button_ids[index],
            targets=[button_targets[index]],
        )
    return level


def build_level_5() -> LevelSpec:
    rng = random.Random(LEVEL_RANDOM_SEED)
    level = level_from_ascii(
        5,
        "Dual Blind Door Control",
        "The symbol * is the goal cell for each goose. Each goose must pass three doors to reach its *. Before each door there are two pressure buttons that affect the other goose's next door while occupied: one opens it and the other does nothing, but which one works is unknown and must be learned through planner-mediated communication.",
        """

.X.$..$..$..*#
..@#.@#.@#...#
..@#.@#.@#...#
##############
#..$..$..$..*#
#..#.@#.@#..@#
#Y.#.@#.@#..@#
##############
""",
    )
    all_doors = _sorted_positions(level, "$")
    all_buttons = _sorted_positions(level, "@")

    upper_doors = _positions_in_row_range(all_doors, 0, 3)
    lower_doors = _positions_in_row_range(all_doors, 4, level.height - 1)
    upper_button_pairs = _paired_positions_by_column(_positions_in_row_range(all_buttons, 0, 3))
    lower_button_pairs = _paired_positions_by_column(_positions_in_row_range(all_buttons, 4, level.height - 1))

    for index, position in enumerate(upper_doors, start=1):
        level.doors[f"g1_door_{index}"] = Door(f"g1_door_{index}", position=position, is_open=False)
    for index, position in enumerate(lower_doors, start=1):
        level.doors[f"g2_door_{index}"] = Door(f"g2_door_{index}", position=position, is_open=False)

    for index in range(3):
        upper_targets = [[f"g2_door_{index + 1}"], []]
        lower_targets = [[f"g1_door_{index + 1}"], []]
        rng.shuffle(upper_targets)
        rng.shuffle(lower_targets)
        upper_pair = upper_button_pairs[index]
        lower_pair = lower_button_pairs[index]
        level.buttons[f"g1_a{index + 1}"] = Button(
            f"g1_a{index + 1}",
            position=upper_pair[0],
            label=f"A{index + 1}",
            targets=upper_targets[0],
        )
        level.buttons[f"g1_b{index + 1}"] = Button(
            f"g1_b{index + 1}",
            position=upper_pair[1],
            label=f"B{index + 1}",
            targets=upper_targets[1],
        )
        level.buttons[f"g2_a{index + 1}"] = Button(
            f"g2_a{index + 1}",
            position=lower_pair[0],
            label=f"A{index + 1}",
            targets=lower_targets[0],
        )
        level.buttons[f"g2_b{index + 1}"] = Button(
            f"g2_b{index + 1}",
            position=lower_pair[1],
            label=f"B{index + 1}",
            targets=lower_targets[1],
        )
    return level


def build_all_levels() -> list[LevelSpec]:
    return [build_level_1(), build_level_2(), build_level_3(), build_level_4(), build_level_5()]
