"""The same task with a local model served by Ollama.

    pip install "corollary[openai]"
    ollama pull llama3.1
    python examples/agent_ollama.py
    python examples/agent_ollama.py llama3.1

The model name can also be set with OLLAMA_MODEL.
"""

from __future__ import annotations

import os
import sys

import openai

from corollary import Agent, OpenAIModel, tool

FILINGS = {"Q1": 4.0e9, "Q2": 4.3e9, "Q3": 4.5e9, "Q4": 4.8e9}


@tool(trust="high")
def get_revenue(quarter: str) -> float:
    """Quarterly revenue in USD from SEC filings. quarter is one of Q1, Q2, Q3, Q4."""
    return FILINGS[quarter]


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OLLAMA_MODEL", "llama3.1")
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    agent = Agent(OpenAIModel(model_name, client=client), tools=[get_revenue])
    report = agent.run("Compare Q2 and Q3 revenue and assess the growth trend.")

    print("ANSWER:", report.answer)
    for rejection in report.rejections:
        print("  rejected:", rejection)
    if not report.completed:
        print("\nThe agent did not answer:", report.error or f"no answer within {agent.max_steps} steps")
        return

    print("\nPROOF:\n" + report.proof.render())
    print("\n" + str(report.verify()))


if __name__ == "__main__":
    main()
