# 🪿 Goose Game Multi-Agent Homework 🪿

## 🎯 Assignment Goal

Your task is to implement a **multi-agent system** that solves Goose Game levels inspired by *Untitled Goose Game*.

In this homework:

- one agent acts as the **planner**,
- two agents act as **goose agents**,
- the planner must coordinate the geese,
- the geese must cooperate through the planner,
- the system must solve levels in a **reproducible** way.

The final solution must work in both available difficulty modes:
- `easy`
- `hard`

The difference between them is visibility:
- in `easy` mode, the whole map is visible,
- in `hard` mode, each goose only sees the part of the map reachable from its current position.

## 🧩 What You Need To Build

You must prepare a solution in which:
- the planner decomposes the problem into tasks,
- the planner sends instructions to the goose agents,
- each goose acts only on the basis of its allowed observations,
- geese return information back to the planner,
- the planner uses these reports to coordinate the team and solve the level.

The purpose of the homework is not only to reach the goal once, but to design a system that solves the tasks in a structured and repeatable way.

## 📌 What Counts As a Correct Solution

Your solution should:
- run correctly in this repository,
- solve the provided game levels,
- handle both `easy` and `hard` mode,
- respect the intended information limitations of the environment,
- use planner-mediated cooperation between agents.

## 🛠️ What You Are Allowed To Modify

You must follow these implementation rules:
- you **cannot modify existing files**, except for `agents/solution.py`,
- you may complete and extend `agents/solution.py`,
- you **must not rename** any existing classes, functions, or identifiers already present in `agents/solution.py`,
- you may add new helper functions in `agents/solution.py`,
- you may add new classes in `agents/solution.py`,
- you may add additional files and directories inside the `agents/` directory.

You should treat the rest of the repository as starter infrastructure for the homework.

## 🚫 Cheating Rules

This homework will be graded strictly.  
If you bypass the intended agent limitations, the solution should receive **0 points**.

Examples of forbidden behavior:
- giving the planner access to the full hidden map,
- reading private or internal environment state that should not be available to the agents,
- hardcoding exact level solutions as direct scripted step lists,
- adding hidden hints that tell agents exactly where to go without discovery,
- bypassing the intended planner-to-goose coordination mechanism.

The intended setup is:
- the planner coordinates,
- each goose knows only what it can observe,
- hidden relations such as button-door dependencies must be discovered through interaction.

## 📦 What You Must Submit

As a final homework submission, you must return:
- a **working implementation**,
- a **final report**.

## 📝 Report Requirements

The report must have:
- at least **1 page** of plain text,
- at most **2 pages** of plain text.

The report should describe:
- your methodology,
- your agent architecture,
- how the planner and geese cooperate,
- how you handled partial observability,
- how you achieved reproducibility,
- what problems you encountered,
- what insights you gained during the homework.

Additional screenshots, logs, or other evidence may help your grade when the assessment is close to the boundary between grades.

## 🤖 Required Model

For this homework, the required model is:
- `google/gemma-4-31b-it:free`

This is already the default model in [settings.py](<settings.py>).

Additional rules:
- the expected host is **OpenRouter**,
- the free endpoint has a limit of **15 requests per minute**,
- the model itself is not intended to be changed or negotiated in the scope of this homework.

You may adapt provider integration details if needed by the implementation, but the required model remains the same.

## 🔐 API Setup

To use the required model through OpenRouter:

1. Create an API key in **Google AI Studio**.
2. Add that Google key to your **OpenRouter** account. [Link to OpenRouter Google integration](https://openrouter.ai/settings/integrations).
3. Generate an **OpenRouter API key**.
4. Store the OpenRouter key in a local `.env` file in the repository root.

This project reads configuration from environment variables via `settings.py`.

## 🧪 Environment Configuration

Create a `.env` file in the repository root.

Example:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemma-4-31b-it:free
```

Why this is required:

- it keeps API keys out of source code,
- it is safer than hardcoding secrets in Python files,
- it matches the configuration expected by the starter code.

## ▶️ How To Run the Homework

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run main.py
```

The interface allows you to:

- choose a level,
- select `easy` or `hard` mode,
- set the action limit,
- run the workflow and inspect communication between planner and geese.

## 🗂️ Repository Structure

```text
nlp-multiagent-homework-goose/
├── main.py                  # Streamlit application used to run and visualize the homework
├── settings.py              # Configuration loaded from .env, including model and OpenRouter settings
├── requirements.txt         # Python dependencies
├── agents/
│   ├── __init__.py
│   ├── base.py              # Base interfaces for planner and goose agents
│   ├── chat_client.py       # OpenAI-compatible client creation for OpenRouter
│   ├── solution.py          # Main file intended for student work
│   └── workflow.py          # Connects the planner, geese, and the environment
└── goose_game/
    ├── __init__.py
    ├── environment.py       # Environment logic, visibility rules, and planner/goose wrappers
    ├── levels.py            # Definitions of all levels used in the assignment
    ├── maps.py              # ASCII map parsing
    ├── models.py            # Data structures for maps, positions, buttons, and doors
    └── render.py            # ASCII rendering and map legend
```

## 🗺️ Levels Included in the Homework

The repository currently contains 5 levels:

1. **Level 1 - Shared Map Final Honk**  
   Both geese must reach the goal cell `*` and honk there.

2. **Level 2 - Button Rescue**  
   One goose must stand on a button to keep a door open for the other goose.

3. **Level 3 - Single Hidden Button**  
   Several buttons are available, but only one opens the required door.

4. **Level 4 - Unknown Door Wiring**  
   Button-door connections are hidden and must be discovered experimentally.

5. **Level 5 - Dual Blind Door Control**  
   Each goose depends on the other goose's actions to progress through the level.

## ⚙️ Important Technical Notes

When working on the homework, remember:

- moves and honks count toward the action budget,
- both geese must finish correctly for the level to be solved,
- the planner should coordinate based on reports, not hidden world knowledge,
- hard mode requires correct handling of partial observability,
- reproducibility is part of the assignment expectation.