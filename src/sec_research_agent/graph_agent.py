"""The same agent built as a LangGraph state machine, to compare with the plain loop in agent.py.

Graph: model -> (tools -> model)* -> END. Uses the same steps as Agent, so results match.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import Agent, Run
from .models import AskResult


class State(TypedDict):
    run: Run
    pending: list[dict[str, Any]]


def build_graph(agent: Agent):
    def model_node(state: State) -> dict[str, Any]:
        return {"pending": agent.model_step(state["run"])}

    def tools_node(state: State) -> dict[str, Any]:
        agent.tool_step(state["run"], state["pending"])
        return {"pending": []}

    def next_step(state: State) -> str:
        run = state["run"]
        if run.final is not None or run.steps >= agent.max_steps:
            return END
        return "tools"

    graph = StateGraph(State)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", next_step, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


def ask_with_graph(agent: Agent, question: str, history: list[tuple[str, str]] | None = None) -> AskResult:
    run = agent.start(question, history)
    if run.final is None:
        graph = build_graph(agent)
        graph.invoke({"run": run, "pending": []}, {"recursion_limit": agent.max_steps * 2 + 2})
    return agent.finish(run)
