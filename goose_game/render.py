from collections.abc import Iterable, Mapping

from goose_game.models import Button, Door, LevelSpec, Position


MAP_LEGEND_MARKDOWN = (
    "`#` blocked  "
    "`.` empty  "
    "`*` goal / final position  "
    "`@` button  "
    "`$` closed door  "
    "`/` open door  "
    "`?` unknown  "
    "`X` goose 1  "
    "`Y` goose 2"
)

EMOJI_TILE_BY_SYMBOL = {
    "#": "🧱",
    ".": "ㆍ",
    "*": "⭐",
    "@": "🔵",
    "$": "⛔",
    "/": "🟢",
    "?": "❔",
    "X": "🪿",
    "Y": "🦢",
}

EMOJI_LEGEND = (
    "🧱 wall  "
    "ㆍ grass  "
    "⭐ goal / final position  "
    "🔵 button  "
    "⛔ closed door  "
    "🟢 open door  "
    "❔ unknown  "
    "🪿 goose 1  "
    "🦢 goose 2"
)


def render_ascii_map(
    spec: LevelSpec,
    goose_positions: Mapping[str, Position],
    buttons: Iterable[Button],
    doors: Iterable[Door],
    visible_positions: set[Position] | None = None,
) -> str:
    button_list = list(buttons)
    door_list = list(doors)
    rows: list[str] = []
    for row in range(spec.height):
        chars: list[str] = []
        for col in range(spec.width):
            position = (row, col)
            if visible_positions is not None and position not in visible_positions:
                char = "?"
            else:
                char = "#"
                if position in spec.walkable_cells:
                    char = "."
                for button in button_list:
                    if button.position == position:
                        char = "@"
                for door in door_list:
                    if door.position == position:
                        char = "/" if door.is_open else "$"
                if goose_positions.get("goose_1") == position:
                    char = "X"
                if goose_positions.get("goose_2") == position:
                    char = "Y"
                if position in spec.goal_cells:
                    char = "*"
            chars.append(char)
        rows.append("".join(chars))
    return "\n".join(rows)


def render_emoji_map(map_text: str) -> str:
    rows = map_text.splitlines()
    emoji_rows: list[str] = []
    for row in rows:
        emoji_rows.append("".join(EMOJI_TILE_BY_SYMBOL.get(char, char) for char in row))
    return "\n".join(emoji_rows)
