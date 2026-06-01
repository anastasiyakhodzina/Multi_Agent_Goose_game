from openai import OpenAI

from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment

import time
from goose_game.models import Direction
import json
import ast


def call_LLM(client: OpenAI, model: str, system: str, user: str, max_retries: int = 10) -> str:
    for attempt in range(max_retries):
        try:
            time.sleep(2)
            print(f"Calling LLM, attempt {attempt + 1}")
            print(f"  system size: {len(system)}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            if not response or not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from API")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(10)
    raise RuntimeError(f"LLM failed after {max_retries} attempts")


def call_LLM_with_tools(client: OpenAI, model: str, system: str, messages: list, tools: list, max_retries: int = 10):
    for attempt in range(max_retries):
        try:
            time.sleep(2)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}] + messages,
                tools=tools,
                parallel_tool_calls=False,
            )
            if not response or not response.choices:
                raise ValueError("Empty response from API")
            return response.choices[0].message
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(10)
    raise RuntimeError(f"LLM failed after {max_retries} attempts")


def extract_last_json(text: str) -> str:
    last = None
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                last = text[start:i + 1]
    return last


def parse_coord(value):
    """Safely turn '(3, 0)' into tuple (3, 0). Returns None on failure or placeholder."""
    if not value:
        return None
    if "<" in str(value) or ">" in str(value):
        return None  # placeholder like <goose_1 pos>
    try:
        result = ast.literal_eval(value)
        if isinstance(result, tuple) and len(result) == 2:
            return result
    except Exception:
        return None
    return None


class SharedBlockedCells:
    """Shared agent memory: cells that must be shown as '.' (blocked) on maps.
    When a goose passes through a door, we block the door cell,
    so it will not pay attention to this door anymore."""

    def __init__(self):
        self.blocked_cells = []  # list of (row, col)

    def add_blocked_cell(self, cell):
        if cell and cell not in self.blocked_cells:
            self.blocked_cells.append(cell)

    def mask_map(self, raw_map: str) -> str:
        lines = [list(line) for line in raw_map.splitlines()]
        for r, c in self.blocked_cells:
            if 0 <= r < len(lines) and 0 <= c < len(lines[r]):
                if lines[r][c] not in ('X', 'Y'):  # never overwrite a goose
                    lines[r][c] = '.'
        return "\n".join("".join(row) for row in lines)


GOOSE_SYSTEM_PROMPT = """You are a goose agent in a grid puzzle.
You receive a phase and a  target position from the planner and must go to it using tools. If you are already standing on target position, do not move.

Map symbols:
`#` blocked, `.` empty, `*` goal, `@` button, `$` closed door, `/` open door, `X` goose1, `Y` goose2

Rules:
- You can only move to: `.` `*` `@` `/` `X` `Y`
- NEVER move into `#` or `$`
- If the next cell is `$` (closed door) — stop and report it, do NOT try to move there
- Honk ONLY when standing on `*` AND the other goose is also on `*`
- Complete your instruction within 10 tool calls maximum

When the planner gives you a target position (e.g. "Your target position: (3, 0)"):
- Compare YOUR current position to the target after each move
- STOP IMMEDIATELY when your position equals the target
- Do not rely on the map symbol — rely on coordinates
- HONK IN HONK PHASE
"""

GOOSE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the goose one step in a direction. Returns success/blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"]}
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "honk",
            "description": "Honk one time in honk phase",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


class GooseAgentImpl(GooseAgent):
    def __init__(self, client: OpenAI, used_model: str, env: GooseEnvironment, append_to_chat: ChatCallback) -> None:
        super().__init__(client, used_model, env, append_to_chat)
        self.positions_history = []
        self._system_prompt = GOOSE_SYSTEM_PROMPT
        self._standing_on = '.'
        self._shared_blocked = None
        self.closest_cells_map = ''

    def _tool_move(self, direction: str) -> str:
        direction_map = {
            'up': Direction.UP, 'down': Direction.DOWN,
            'left': Direction.LEFT, 'right': Direction.RIGHT
        }
        map_before = self._env.describe_state()
        if self._shared_blocked is not None:
            map_before = self._shared_blocked.mask_map(map_before)
        map_before = map_before.splitlines()

        cur_position = self._env.visible_goose_positions().get(self._env.goose_id)
        row_change, column_change = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}[direction]
        row, col = cur_position[0] + row_change, cur_position[1] + column_change

        symbol = '?'
        if 0 <= row < len(map_before) and 0 <= col < len(map_before[row]):
            symbol = map_before[row][col]

        success = self._env.move(direction_map[direction])
        if success:
            self._standing_on = symbol

        geese_positions = self._env.visible_goose_positions()
        current_position = geese_positions.get(self._env.goose_id, 'unknown')
        return f"{'success' if success else 'failed'}. My position {current_position}, standing on '{self._standing_on}'."

    def _tool_honk(self) -> str:
        try:
            self._env.honk(count=1)
            geese_positions = self._env.visible_goose_positions()
            current_position = geese_positions.get(self._env.goose_id, 'unknown')
            return f" success. My position {current_position}."
        except RuntimeError:
            return 'failed. reached actions limit'

    def _numbered_map(self) -> str:
        raw_map = self._env.describe_state()
        if self._shared_blocked is not None:
            raw_map = self._shared_blocked.mask_map(raw_map)
        lines = raw_map.splitlines()
        col_count = len(lines[0]) if lines else 0
        header = "col: " + "".join(str(i % 10) for i in range(col_count))
        rows = [f"r{i:2}: {line}" for i, line in enumerate(lines)]
        return "\n".join([header] + rows)

    def _closest_cells_map(self, current_position: tuple = None) -> str:
        """5x5 square (radius 2) around the goose, with real row/col numbers."""
        if current_position is None or current_position == 'unknown':
            return "position unknown"
        raw_map = self._env.describe_state()
        if self._shared_blocked is not None:
            raw_map = self._shared_blocked.mask_map(raw_map)
        current_row, current_column = current_position
        lines = raw_map.splitlines()
        num_rows = len(lines)
        num_columns = len(lines[0])
        rows = [r for r in range(current_row - 2, current_row + 3) if 0 <= r < num_rows]
        columns = [c for c in range(current_column - 2, current_column + 3) if 0 <= c < num_columns]
        header = 'col: ' + ''.join(str(c % 10) for c in columns)
        result = [header]
        for r in rows:
            row = [lines[r][c] for c in columns]
            result.append(f"r{r:2}: {''.join(row)}")
        return "\n".join(result)

    def set_shared_blocked(self, class_instance):
        self._shared_blocked = class_instance

    def on_call(self, message: GooseAgentMessage) -> GooseAgentResult:
        self._append_to_chat(f"Planner message: {message.description}")

        visible_map = self._numbered_map()
        geese_positions = self._env.visible_goose_positions()
        current_position = geese_positions.get(self._env.goose_id, 'unknown')
        self.closest_cells_map = self._closest_cells_map(current_position)

        desc_lower = message.description.lower()
        is_own_hold = (
            (self._env.goose_id == "goose_1" and "phase: goose1_hold" in desc_lower) or
            (self._env.goose_id == "goose_2" and "phase: goose2_hold" in desc_lower)
        )

        # skip LLM in start / own hold / observe phases, do not move, just observe
        if ("phase: start" in desc_lower) or is_own_hold or ('goose_1 observe' in desc_lower):
            self.positions_history.append(current_position)
            goose_info = f"""
            My current position: {current_position}, standing on {self._standing_on}
            Visible geese positions: {geese_positions}
            My last 3 positions: {self.positions_history[-3:]}
            Closest cells map:
{self.closest_cells_map}
            Visible map:
{visible_map}
            """
            return GooseAgentResult(output=goose_info)

        messages = [{
            "role": "user",
            "content": f"""Instruction: {message.description}

        My position: {current_position}, standing on {self._standing_on}
        Visible positions: {geese_positions}
        Visible map:
{visible_map}

        Execute your instruction using tools. Max 10 tool calls."""
        }]

        tool_mapping = {"move": self._tool_move, "honk": self._tool_honk}

        for iteration in range(10):
            llm_answer = call_LLM_with_tools(self._client, self._used_model, self._system_prompt, messages, GOOSE_TOOLS)
            messages.append(llm_answer)
            if llm_answer.content:
                self._append_to_chat(f"goose llm answer: {str(llm_answer.content)}")
            if not llm_answer.tool_calls:
                self._append_to_chat(f"Finished after {iteration + 1} iterations")
                break
            for tool_call in llm_answer.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                result = tool_mapping[name](**args)
                direction = args.get('direction')
                if direction:
                    self._append_to_chat(f"{direction} finished with {result[:28]}")
                else:
                    self._append_to_chat(f"{name} finished with {result[:28]}")
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        visible_map = self._numbered_map()
        geese_positions = self._env.visible_goose_positions()
        current_position = geese_positions.get(self._env.goose_id, 'unknown')
        self.positions_history.append(current_position)
        self.closest_cells_map = self._closest_cells_map(current_position)

        if self._env.goose_id == "goose_1":
            # goose_1 omits the full visible_map to save tokens
            goose_info = f"""
            My current position: {current_position}, standing on {self._standing_on}
            Geese (which I see) positions: {geese_positions}
            My last 3 positions: {self.positions_history[-3:]}
            """
        else:
            goose_info = f"""
            My current position: {current_position}, standing on {self._standing_on}
            Geese (which I see) positions: {geese_positions}
            My last 3 positions: {self.positions_history[-3:]}
            Closest cells map:
{self.closest_cells_map}
            Visible map:
{visible_map}
            """

        self._append_to_chat('After completing planner task I have :')
        self._append_to_chat(goose_info)
        return GooseAgentResult(output=goose_info)


PLANNER_SYSTEM_PROMPT = """You are a planner coordinating two geese (goose_1 and goose_2) in a grid puzzle.
Both geese must reach * and honk there together.

Map symbols: `#` blocked, `.` empty, `*` goal, `@` button, `$` closed door, `/` open door, `X` goose1, `Y` goose2
Geese can only move to: `.` `*` `@` `/` — NOT to `#` or `$`

READING COORDINATES (row, col), starting from 0:
- Row = line index from top (first line = row 0)
- Column = character index in line (first char = col 0)
- Example: if row 0 is ".*..#", then * is at col 1, NOT col 0
- ALWAYS count every character including dots before the symbol

When a goose stands ON a button or goal, the goose symbol (X/Y) is shown instead of @ or *.
Use position reports to know where geese are.

Respond with ONLY a JSON object:
{
  "reasoning": "brief explanation",
  "next_phase": "goose1_hold",
  "goal_position": "(0,7)",
  "tried_button_position": "(3,0)",
  "target_position_goose_1": "(3, 0)",
  "target_position_goose_2": "(2, 5)",
  "door_to_check": "(4, 7)",
  "blocked_cell": "(4, 8)"
}

GENERAL RULES:
- Goal position never changes — once found, always return the same value.
- Both geese must be on * at the same time before honking.
- The user prompt contains PHASE-SPECIFIC INSTRUCTIONS — follow them carefully for the current phase.

Level description:
{level_description}
"""


class PlannerAgentImpl(PlannerAgent):
    def __init__(self, client, used_model, env, agents, append_to_chat) -> None:
        super().__init__(client, used_model, env, agents, append_to_chat)
        self._phase = "start"
        self._system_prompt = PLANNER_SYSTEM_PROMPT.replace("{level_description}", self._env.task_description)
        self._goal_position = None
        self._last_reasoning = ""

        self._completed_buttons = []        # buttons that WORKED (door opened + goose passed). Never reuse.
        self._tested_for_current_door = []  # buttons tried for the CURRENT door. Reset when a door opens.
        self.target_geese_positions = {"goose_1": None, "goose_2": None}

        self._door_to_check = None          # door we check for '/' in hold phase

        self._shared_blocked = SharedBlockedCells()
        for goose in self._agents.values():
            goose.set_shared_blocked(self._shared_blocked)

        self._passed_doors = []

    def _get_phase_instruction(self) -> str:
        known = ["start", "goose1_move_to_button", "goose1_move_to_button_check", "goose1_hold", "goose1_hold_hold",
                 "goose2_move_to_button", "goose2_move_to_button_check", "goose2_hold", "goose2_hold_hold",
                 "both_to_goal", "honk"]
        if self._phase not in known:
            self._phase = "start"

        if self._phase == "start":
            return """
- This is the initial step. Look at both maps to find:
  - Goal positions * (record in goal_position)(there might be two goals *, record both then).
  - Button positions @.
  - Door positions $.
- If you see `@` or `$` on the maps:
  - If goose_2 is closer to buttons -> next_phase = "goose2_move_to_button", "target_position_goose_2" = button position, "target_position_goose_1" = cell ADJACENT to the closed door `$` that is closest to goose_1 position.Record that door in "door_to_check".
  - If goose_1 is closer to buttons -> next_phase = "goose1_move_to_button", "target_position_goose_1" = button position, "target_position_goose_2" = cell ADJACENT to the closed door `$` that is closest to goose_2 position.Record that door in "door_to_check".
- If you see NO `@` and NO `$` -> next_phase = "both_to_goal".
"""

        if self._phase == "goose2_move_to_button":
            return """
- If goose_2 sees NO untested button in ITS CLOSEST CELLS MAP!!! AND is not standing on one -> next_phase = start !!
  else:
    - CHECK target_position_goose_2 if it is on the untested button within the CLOSEST CELLS MAP of goose_2, if it is not change
    - LOOK at goose_1's CLOSEST CELLS MAP!! Find the door (`$` or `/`) closest to goose_1 ON CLOSEST CELLS MAP, NEVER pick a door that is in the "Passed doors" list 
      Record it in "door_to_check". SET target_position_goose_1 to a cell adjacent to that door and closest to goose_1 (it might be goose_1 position, if goose_1 already standing in the next cell to '$' or '/' cell)
      (up/down/left/right) and closest to goose_1.
      - If goose_2 is ON the button -> next_phase = "goose2_move_to_button_check", 
        else  next_phase = "goose2_move_to_button".
        
- Do NOT include tried_button_position or blocked_cell here.
"""

        if self._phase == "goose2_move_to_button_check":
                return """
        - KEEP target_position_goose_2.
        -  CHECK 'door_to_check' position:
            LOOK at goose_1's CLOSEST CELLS MAP!! Find the door (`$` or `/`) closest to goose_1 ON CLOSEST CELLS MAP, NEVER pick a door that is in the "Passed doors" list 
            CHECK if it in "door_to_check", if not change 'door_to_check' to this door ('$' or '/') position. Set target_position_goose_1 to a cell adjacent to that door
          (up/down/left/right) and closest to goose_1 (it might be goose_1 position, if goose_1 already standing in the next cell to '$' or '/' cell). IF GOOSE_1 STANDING ON '/' or '$', CHANGE GOOSE_1 POSITION ACCORDING TO PREVIOUS RULES!!!
          - If goose_2 is ON the button -> next_phase = "goose2_hold", 
            else  next_phase = "goose2_move_to_button_check" and keep "target_position_goose_2" and keep "target_position_goose_1".
        - Do NOT include tried_button_position or blocked_cell here.
        """

        if self._phase == "goose1_move_to_button":
            return """
- If goose_1 sees NO untested button in ITS CLOSEST CELLS MAP!!! AND is not standing on one -> next_phase = start !!
  else:
    - CHECK target_position_goose_1 if it is on the untested button within the CLOSEST CELLS MAP of goose_1, if it is not change
    - LOOK at goose_2's CLOSEST CELLS MAP!! Find the door (`$` or `/`) closest to goose_2 ON CLOSEST CELLS MAP, NEVER pick a door that is in the "Passed doors" list 
      Record it in "door_to_check". SET target_position_goose_2 to a cell adjacent to that door and closest to goose_2 (it might be goose_2 position, if goose_2 already standing in the next cell to '$' or '/' cell)
      (up/down/left/right) and closest to goose_2.
      - If goose_1 is ON the button -> next_phase = "goose1_move_to_button_check", 
        else  next_phase = "goose1_move_to_button".
        
- Do NOT include tried_button_position or blocked_cell here.
"""

        if self._phase == "goose1_move_to_button_check":
                return """
        - KEEP target_position_goose_1.
        -  CHECK 'door_to_check' position:
            LOOK at goose_2's CLOSEST CELLS MAP!! Find the door (`$` or `/`) closest to goose2 ON CLOSEST CELLS MAP, NEVER pick a door that is in the "Passed doors" list 
            CHECK if it in "door_to_check", if not change 'door_to_check' to this door ('$' or '/') position. Set target_position_goose_2 to a cell adjacent to that door
          (up/down/left/right) and closest to goose_2 (it might be goose_2 position, if goose_2 already standing in the next cell to '$' or '/' cell). IF GOOSE_2 STANDING ON '/' or '$', CHANGE GOOSE_2 POSITION ACCORDING TO PREVIOUS RULES!!!
          - If goose_1 is ON the button -> next_phase = "goose1_hold", 
            else  next_phase = "goose1_move_to_button_check" and keep "target_position_goose_2" and keep "target_position_goose_1".
        - Do NOT include tried_button_position or blocked_cell here.
        """

        if self._phase == "goose2_hold":
            return """
Look at the EXACT coordinate of door_to_check on the maps.

CASE A: door_to_check shows `$` or '.' or '#' -> this button does NOT open it.
- RETURN tried_button_position = goose_2's current position !
- Look at goose_2's Closest cells map. Are there any `@` buttons NOT in 
  completed_buttons and NOT in tested_for_current_door?
- If at goose_2's CLOSEST CELLS MAP you SEE button '@' which IS NOT in completed_buttons and NOT in tested_for_current_door then : 
    -target_position_goose_2 = that button.
    -next_phase = "goose2_move_to_button".
    -Keep door_to_check the same. KEEP target_position_goose_1.
- If at goose_2's CLOSEST CELLS MAP you DO NOT see button '@' which IS NOT in completed_buttons and NOT in tested_for_current_door then :
    -next_phase = "goose1_move_to_button"!!!!
    -target_position_goose_1 = untested button '@' at goose_1's CLOSEST CELLS MAP!!!! which IS NOT in completed_buttons and NOT in tested_for_current_door and has THE BIGGEST SECOND COORDINATE(the most right)
    -Keep target_position_goose_2 and door_to_check ( in the next phase change them within instructions)
- Do NOT include blocked_cell.

CASE B: door_to_check shows `/` -> this button WORKS!
- target_position_goose_2 = goose_2's current position.
- Set target_position_goose_1 = the cell on the FAR side of the door:
  look at the door's 4 neighbors, pick the one marked `.` that is NOT where goose_1 stands now.
  (goose_1 will step onto the door `/` then onto that `.` cell.)
- next_phase = "goose2_hold_hold".
- Do NOT include tried_button_position or blocked_cell here.
"""

        if self._phase == "goose1_hold":
            return """
Look at the EXACT coordinate of door_to_check on the maps.

CASE A: door_to_check shows `$` or '.' or '#' -> this button does NOT open it.
- RETURN tried_button_position = goose_1's current position!
- Look at goose_1's Closest cells map. Are there any `@` buttons NOT in 
  completed_buttons and NOT in tested_for_current_door?
  - YES: 
    target_position_goose_1 = that button.
    next_phase = "goose1_move_to_button".
    Keep door_to_check the same. KEEP target_position_goose_2.
  - NO:
    Now goose_2 should try ITS nearby buttons.
    target_position_goose_2 = untested button '@' at goose_2's CLOSEST CELLS MAP!!!! which IS NOT in completed_buttons and NOT in tested_for_current_door and has THE BIGGEST SECOND COORDINATE(the most right)
    Keep target_position_goose_1 and door_to_check ( in the next phase change them within instructions)
    next_phase = "goose2_move_to_button".
- Do NOT include or blocked_cell.

CASE B: door_to_check shows `/` -> this button WORKS!
- target_position_goose_1 = goose_1's current position.
- Set target_position_goose_2 = the cell on the FAR side of the door:
  look at the door's 4 neighbors, pick the one marked `.` that is NOT where goose_2 stands now.
  (goose_2 will step onto the door `/` then onto that `.` cell.)
- next_phase = "goose1_hold_hold".
- Do NOT include tried_button_position or blocked_cell here.
"""

        if self._phase == "goose2_hold_hold":
            return """
- Keep target_position_goose_2 
- Keep target_position_goose_1
- Check goose_1's position:
  CASE PASSED: current goose_1 position is equal to target_position_goose_1:
      - Set "blocked_cell" = door_to_check. It will become '.' on the map and should NEVER be picked again.
      - Check ONLY cells DIRECTLY adjacent to each goose (up/down/left/right, distance 1):
        - Is there a `$` in those 4 cells of goose_1? Is there a `$` in those 4 cells of goose_2?
        - IGNORE all `$` that are further away — they do NOT matter.
      - If a `$` is in the 4 adjacent cells of a goose_1 (ONLY check NEXT left, right, up, down cells with DISTANCE 1) -> next_phase = "goose2_move_to_button" and target_position_goose_2 = the most right untested button in the CLOSEST CELLS MAP of goose_2. 
      - If a `$` is in the 4 adjacent cells of a goose_2 (ONLY check NEXT left, right, up, down cells with DISTANCE 1)-> next_phase = "goose1_move_to_button" and target_position_goose_1 = the most right untested button in the CLOSEST CELLS MAP of goose_1. 
      - If NO `$` in the 4 adjacent cells of EITHER goose -> next_phase = "both_to_goal", target_position_goose_1 = goal_position and target_position_goose_2 = goal_position .
        Do NOT try to open doors that are not adjacent. 
  CASE NOT YET: current goose_1 position is not equal to target_position_goose_1  ->  next_phase = "goose2_hold_hold", keep target_position_goose_1  and target_position_goose_2. Do NOT set blocked_cell.
"""

        if self._phase == "goose1_hold_hold":
            return """
- Keep target_position_goose_2 
- Keep target_position_goose_1
- Check goose_2's position:
  CASE PASSED: current goose_2 position is equal to target_position_goose_2:
      - Set "blocked_cell" = door_to_check. It will become '.' on the map and should NEVER be picked again.
      - Check ONLY cells DIRECTLY adjacent to each goose (up/down/left/right, distance 1):
        - Is there a `$` in those 4 cells of goose_1? Is there a `$` in those 4 cells of goose_2?
        - IGNORE all `$` that are further away — they do NOT matter.
      - If a `$` is in the 4 adjacent cells of a goose_1 (ONLY check NEXT left, right, up, down cells with DISTANCE 1)-> next_phase = "goose2_move_to_button" and target_position_goose_2 = the most right untested button in the CLOSEST CELLS MAP of goose_2. 
      - If a `$` is in the 4 adjacent cells of a goose_2 (ONLY check NEXT left, right, up, down cells with DISTANCE 1)-> next_phase = "goose1_move_to_button" and target_position_goose_1 = the most right untested button in the CLOSEST CELLS MAP of goose_1.
      - If NO `$` in the 4 adjacent cells of EITHER goose -> next_phase = "both_to_goal", target_position_goose_1 = goal_position and target_position_goose_2 = goal_position .
        Do NOT try to open doors that are not adjacent. 
  CASE NOT YET: current goose_2 position is not equal to target_position_goose_2  ->  next_phase = "goose1_hold_hold", keep target_position_goose_1  and target_position_goose_2. Do NOT set blocked_cell.
"""

        if self._phase == "both_to_goal":
            return """ 
- Check ONLY the 4 cells DIRECTLY adjacent (distance 1: up/down/left/right) to each goose for a `$`.
  IGNORE EVERY `$` that is FURTHER away than 1 CELL from each goose!!!!!!!!!!!!
  Example: goose at (4,2). Adjacent cells are (3,2), (5,2), (4,1), (4,3). A `$` at (4,5) is distance 3 -> IGNORE.
  - If a `$` is in the 4 ADJACENT CELLS!! (up/left/right/down) of a goose  OR YOU DO NOT SEE GOAL *-> that goose is blocked, a door still needs opening.
    Send the goose closer to buttons to a button (not in completed_buttons),
    the other goose next to that adjacent `$`, record door_to_check,
    next_phase = "goose1_move_to_button" or "goose2_move_to_button".
  - If NO `$` is in the 4 ADJACENT CELLS!! (up/left/right/down)  of EITHER goose -> each goose move to the its CLOSEST goal.
    Do NOT open far-away doors; many lead to dead ends and are not needed.
    - target_position_goose_1 = goal closest to goose_1
    - target_position_goose_2 = goal closest to goose_2
    - If both geese ARE on the goal -> next_phase = "honk" !!!
    - else keep next_phase = "both_to_goal".
"""

        if self._phase == "honk":
            return """
- Both geese honk at the goal.
- Keep both targets = goal_position.
"""
        return ""

    def step(self) -> None:
        geese_info = []
        for goose_id, goose in sorted(self._agents.items()):
            description = f"Current phase: {self._phase}. Target position: {self.target_geese_positions[goose_id]}."
            task = GooseAgentMessage(description=description)
            self._append_to_chat(f"Calling {goose_id} with phase: '{self._phase}'")
            result = goose.on_call(task)
            geese_info.append(result)

        description = f"Current phase: goose_1 observe. Target position: None." # after goose_2 moves visible goose_1 map might has changed, so we say goose_1 only to give observed map
        task = GooseAgentMessage(description=description)
        result = self._agents["goose_1"].on_call(task)
        geese_info.append(result)


        goose_info1 = geese_info[2].output
        goose_info2 = geese_info[1].output
        phase_instruction = self._get_phase_instruction()

        user_prompt = f"""
        Current phase: {self._phase}
        Known goals position: {self._goal_position if self._goal_position else 'not found yet - look for * on map'}
        Buttons that WORKED (completed_buttons, never reuse): {self._completed_buttons}
        Buttons already tried for the current door (tested_for_current_door, skip these if needed): {self._tested_for_current_door}
        Door to check: {self._door_to_check}

        Goose 1 info:
        {goose_info1}
        Goose 1 previous target position: {self.target_geese_positions['goose_1']}

        Goose 2 info:
        {goose_info2}
        Goose 2 previous target position: {self.target_geese_positions['goose_2']}

        PHASE-SPECIFIC INSTRUCTIONS FOR "{self._phase}":
        {phase_instruction}
        
        NEVER PICK THIS COORDINATES as door_to_check: {self._passed_doors}  !!!!!!!!!!!!!!!!!
        """
        self._append_to_chat(f"PROMPT SIZE: system={len(self._system_prompt)}, user={len(user_prompt)}")
        print('raw planner user prompt', user_prompt)
        raw = call_LLM(self._client, self._used_model, self._system_prompt, user_prompt)

        raw_clean = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        raw_clean = extract_last_json(raw_clean)
        if raw_clean is None:
            self._append_to_chat("Planner LLM returned no JSON, skipping step")
            return
        try:
            parsed = json.loads(raw_clean)
        except json.JSONDecodeError as e:
            self._append_to_chat(f"Planner JSON parse error: {e}, skipping")
            return
        if not isinstance(parsed, dict):
            self._append_to_chat(f"Planner got non-dict JSON, skipping")
            return

        self._append_to_chat(f"Planner raw:{str(raw)}")
        print(f"Planner raw:{str(raw)}")

        # remember which phase we are LEAVING (to apply correct side effects)
        leaving_phase = self._phase

        self._last_reasoning = parsed.get("reasoning", "")

        # goal lock
        goal_pos = parsed.get("goal_position", "")
        if goal_pos and self._goal_position is None:
            self._goal_position = goal_pos
            self._append_to_chat(f"Goal locked at: {self._goal_position}")

        # tried button (non-working) -> tested_for_current_door, NOT completed
        tried = parse_coord(parsed.get("tried_button_position", ""))
        if tried and tried not in self._tested_for_current_door and tried not in self._completed_buttons:
            self._tested_for_current_door.append(tried)
            self._append_to_chat(f"Button did not work, tested_for_current_door: {self._tested_for_current_door}")

        # door_to_check
        door = parse_coord(parsed.get("door_to_check", ""))
        if door:
            self._door_to_check = door

        # blocked_cell (set in hold_hold after goose passed) -> mask + complete the button
        passed = parse_coord(parsed.get("blocked_cell", ""))
        if passed and leaving_phase in ("goose1_hold_hold", "goose2_hold_hold"):
            self._shared_blocked.add_blocked_cell(passed)
            self._passed_doors.append(passed)
            self._append_to_chat(f"Door passed: {passed}")

             #when we are in phase _hold_hold , write goose position as completed_buttons
            holding = self.target_geese_positions['goose_2'] if leaving_phase == "goose2_hold_hold" else \
            self.target_geese_positions['goose_1']
            holding_coord = parse_coord(holding)
            if holding_coord and holding_coord not in self._completed_buttons:
                self._completed_buttons.append(holding_coord)
            self._tested_for_current_door = []

        # phase + targets
        self._phase = parsed.get("next_phase", self._phase)
        self.target_geese_positions = {
            "goose_1": parsed.get("target_position_goose_1", ""),
            "goose_2": parsed.get("target_position_goose_2", "")
        }