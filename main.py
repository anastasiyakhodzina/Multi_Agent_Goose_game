from html import escape

import streamlit as st

from agents.base import ChatCallback
from agents.chat_client import create_chat_client
from agents.solution import GooseAgentImpl, PlannerAgentImpl
from agents.workflow import LevelWorkflow
from openai import OpenAI
from goose_game.environment import GooseGameEnvironment
from goose_game.levels import (
    build_level_1,
    build_level_2,
    build_level_3,
    build_level_4,
    build_level_5,
)
from goose_game.models import LevelSpec
from goose_game.render import EMOJI_LEGEND, render_ascii_map, render_emoji_map


LEVEL_BUILDERS = {
    "Level 1": build_level_1,
    "Level 2": build_level_2,
    "Level 3": build_level_3,
    "Level 4": build_level_4,
    "Level 5": build_level_5,
}


def _normalize_log_line(message: str) -> str:
    return message.replace("\\n", "\n")


def render_chat(placeholder: st.delta_generator.DeltaGenerator, lines: list[str], title: str) -> None:
    content = "\n\n".join(_normalize_log_line(line) for line in lines) if lines else "No messages yet."
    escaped_content = escape(content)
    html = f"""
    <div style="border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem; padding: 0.75rem; background-color: rgba(255, 255, 255, 0.1); ">
      <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">{escape(title)}</div>
      <div style="height: 320px; margin-top: 0.35rem; overflow-y: auto; white-space: pre-wrap; font-family: monospace; font-size: 0.875rem; line-height: 1.4;">{escaped_content}</div>
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)


def render_centered_page_heading(title: str, caption: str) -> None:
    html = f"""
    <div style="text-align: center; margin-bottom: 1.25rem;">
      <h1 style="margin: 0;">{escape(title)}</h1>
      <div style="margin-top: 0.35rem; color: rgba(250, 250, 250, 0.92);">{escape(caption)}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_centered_section_heading(title: str) -> None:
    st.markdown(
        f'<h3 style="text-align: center; margin-bottom: 1rem;">{escape(title)}</h3>',
        unsafe_allow_html=True,
    )


def render_map_with_legend(placeholder: st.delta_generator.DeltaGenerator, map_text: str, legend_markdown: str) -> None:
    rows = map_text.splitlines()
    if not rows:
        rows = ["🟩"]

    grid_rows = []
    for row in rows:
        cell_html = "".join(
            f'<span style="width: 2rem; height: 2rem; display: inline-flex; align-items: center; justify-content: center; font-size: 1.45rem; line-height: 1;">{escape(char)}</span>'
            for char in row
        )
        grid_rows.append(
            '<div style="display: flex; justify-content: center; gap: 0;">'
            f"{cell_html}"
            "</div>"
        )

    legend_rows = []
    for item in (part.strip() for part in legend_markdown.split("  ") if part.strip()):
        symbol, _, description = item.partition(" ")
        legend_rows.append(
            '<div style="line-height: 1.6; display: flex; align-items: center; gap: 0.45rem;">'
            f'<span style="font-size: 1.1rem; line-height: 1;">{escape(symbol)}</span>'
            f"<span>{escape(description)}</span>"
            "</div>"
        )

    html = f"""
    <div style="display: flex; justify-content: center; width: 100%; margin-bottom: 1.25rem;">
      <div style="width: min(100%, 54rem); display: flex; justify-content: center; align-items: flex-start; gap: 1.25rem; flex-wrap: wrap;">
        <div style="display: inline-flex; flex-direction: column; align-items: center; border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem; padding: 1rem; background-color: rgba(255, 255, 255, 0.05); font-weight: 600; line-height: 1; white-space: nowrap;">
          {''.join(grid_rows)}
        </div>
        <div style="min-width: 18rem; max-width: 24rem; padding-top: 0.25rem;">
          <div style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.35rem;">Legend</div>
          <div style="display: grid; grid-template-columns: repeat(2, minmax(8.5rem, 1fr)); column-gap: 1.25rem; row-gap: 0.1rem;">
            {''.join(legend_rows)}
          </div>
        </div>
      </div>
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)


def render_level_description(placeholder: st.delta_generator.DeltaGenerator, spec: LevelSpec) -> None:
    html = f"""
    <div style="display: flex; justify-content: center; width: 100%; margin-bottom: 0.75rem;">
      <div style="width: min(100%, 54rem); padding: 0.85rem 1rem; text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.35rem;">{escape(spec.name)}</div>
        <div style="font-size: 0.95rem; line-height: 1.4;">{escape(spec.task_description)}</div>
      </div>
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)


def create_workflow(
    level_name: str,
    easy_mode: bool,
    client: OpenAI,
    used_model: str,
    max_turns: int,
    planner_append_to_chat: ChatCallback,
    goose_append_to_chat_by_id: dict[str, ChatCallback],
) -> LevelWorkflow:
    return LevelWorkflow(
        spec=LEVEL_BUILDERS[level_name](),
        easy_mode=easy_mode,
        client=client,
        used_model=used_model,
        planner_factory=PlannerAgentImpl,
        goose_factory=GooseAgentImpl,
        max_turns=max_turns,
        planner_append_to_chat=planner_append_to_chat,
        goose_append_to_chat_by_id=goose_append_to_chat_by_id,
    )


def render_map(workflow: LevelWorkflow | None, spec: LevelSpec, easy_mode: bool) -> None:
    render_level_description(level_description_placeholder, spec)
    if workflow is None:
        ascii_map = render_ascii_map(
            spec,
            spec.goose_starts,
            spec.buttons.values(),
            spec.doors.values(),
            visible_positions=None if easy_mode else GooseGameEnvironment(spec, easy_mode=False).visible_positions_for_display(),
        )
    else:
        ascii_map = render_ascii_map(
            workflow.spec,
            workflow.env.state.goose_positions,
            workflow.env.state.buttons.values(),
            workflow.env.state.doors.values(),
            visible_positions=workflow.env.visible_positions_for_display(),
        )
    render_map_with_legend(map_placeholder, render_emoji_map(ascii_map), EMOJI_LEGEND)


def render_status(
    placeholder: st.delta_generator.DeltaGenerator,
    turn_count: int,
    max_turns: int,
    solved: bool | None = None,
) -> None:
    if solved is True:
        placeholder.success(f"Solved in {turn_count} turns.")
        return
    if solved is False:
        placeholder.warning(f"Not solved in {turn_count} turns (limit: {max_turns}).")
        return
    placeholder.info(f"Turns used: {turn_count}/{max_turns}")


st.set_page_config(page_title="Goose Game", layout="wide")
render_centered_page_heading(
    "Goose Game",
    "Choose a level, create agents for that level, and run planner turns.",
)

with st.sidebar:
    st.header("Menu")
    level_name = st.selectbox("Level", list(LEVEL_BUILDERS))
    easy_mode = st.checkbox("Easy mode", value=True)
    max_turns = st.number_input("Max counted turns", min_value=1, max_value=1000, value=200)
    run_clicked = st.button("Run workflow", type="primary")

level_preview_spec = LEVEL_BUILDERS[level_name]()
level_description_placeholder = st.empty()
map_placeholder = st.empty()
status_placeholder = st.empty()
summary_placeholder = st.empty()
chat_columns = st.columns(3)
planner_chat_placeholder = chat_columns[0].empty()
goose_1_chat_placeholder = chat_columns[1].empty()
goose_2_chat_placeholder = chat_columns[2].empty()

render_map(None, level_preview_spec, easy_mode)
render_status(status_placeholder, 0, int(max_turns))
render_chat(planner_chat_placeholder, [], "Planner")
render_chat(goose_1_chat_placeholder, [], "Goose 1")
render_chat(goose_2_chat_placeholder, [], "Goose 2")

if run_clicked:
    try:
        client, used_model = create_chat_client()
        planner_chat: list[str] = []
        goose_1_chat: list[str] = []
        goose_2_chat: list[str] = []

        def append_planner(message: str) -> None:
            planner_chat.append(message)
            render_chat(planner_chat_placeholder, planner_chat, "Planner")

        def append_goose_1(message: str) -> None:
            goose_1_chat.append(message)
            render_chat(goose_1_chat_placeholder, goose_1_chat, "Goose 1")

        def append_goose_2(message: str) -> None:
            goose_2_chat.append(message)
            render_chat(goose_2_chat_placeholder, goose_2_chat, "Goose 2")

        workflow = create_workflow(
            level_name,
            easy_mode,
            client,
            used_model,
            int(max_turns),
            planner_append_to_chat=append_planner,
            goose_append_to_chat_by_id={
                "goose_1": append_goose_1,
                "goose_2": append_goose_2,
            },
        )
        workflow.env.register_action_callback(
            lambda: (
                render_map(workflow, workflow.spec, easy_mode),
                render_status(status_placeholder, workflow.env.counted_actions, int(max_turns)),
            )
        )
        render_map(workflow, workflow.spec, easy_mode)
        render_status(status_placeholder, workflow.env.counted_actions, int(max_turns))
        solved, turn_count = workflow.run()
        render_map(workflow, workflow.spec, easy_mode)
        render_status(status_placeholder, turn_count, int(max_turns), solved=solved)
    except Exception as exc:  # noqa: BLE001
        status_placeholder.error(str(exc))
