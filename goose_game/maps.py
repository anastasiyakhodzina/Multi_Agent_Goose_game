from dataclasses import dataclass, field
from textwrap import dedent

from goose_game.models import Position


@dataclass(slots=True)
class ParsedMap:
    width: int
    height: int
    walkable_cells: set[Position]
    goose_starts: dict[str, Position]
    special_cells: dict[str, list[Position]] = field(default_factory=dict)


def parse_ascii_map(map_text: str) -> ParsedMap:
    normalized = dedent(map_text).strip()
    lines = [line for line in normalized.splitlines() if line]
    height = len(lines)
    width = max(len(line) for line in lines)
    walkable_cells: set[Position] = set()
    goose_starts: dict[str, Position] = {}
    special_cells: dict[str, list[Position]] = {
        "*": [],
        "@": [],
        "$": [],
        "/": [],
        "?": [],
        "R": [],
        "G": [],
        "B": [],
    }

    for row, line in enumerate(lines):
        if len(line) != width:
            raise ValueError("All map rows must have the same width.")
        for col, cell in enumerate(line):
            position = (row, col)
            if cell == "#":
                continue
            if cell not in {"#", ".", "X", "Y", "*", "@", "$", "/", "?", "R", "G", "B"}:
                raise ValueError(f"Unsupported map symbol: {cell}")
            walkable_cells.add(position)
            if cell == "X":
                goose_starts["goose_1"] = position
            elif cell == "Y":
                goose_starts["goose_2"] = position
            elif cell in special_cells:
                special_cells[cell].append(position)

    if set(goose_starts) != {"goose_1", "goose_2"}:
        raise ValueError("Each map must contain both X and Y start positions.")

    return ParsedMap(
        width=width,
        height=height,
        walkable_cells=walkable_cells,
        goose_starts=goose_starts,
        special_cells=special_cells,
    )
