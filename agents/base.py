# agents/base.py
# Base class for all Mnemo-compatible agents.
# Reference agents inherit from this.
# Clients building new agents from scratch can also inherit from this.
# Clients with existing agents don't need this at all — they register directly.

from abc import ABC, abstractmethod
from typing import Any


class MnemoAgent(ABC):
    """
    Minimum interface every Mnemo-compatible agent must satisfy.

    Attributes
    ----------
    name : str
        Unique identifier for this agent. Used in logs, reputation
        scores, and memory records. Should be descriptive.
        Example: "researcher-llama3", "writer-mistral"

    description : str
        One sentence describing what this agent does.
        Used by the judge to understand what it is evaluating
        and by the router to match agents to task types.

    tools : list
        Tools this agent has access to. Web search, database
        connections, APIs etc. Defaults to empty list.
        Mnemo logs tool usage in every interaction record.
    """

    def __init__(self, name: str, description: str, tools: list = None):
        self.name = name
        self.description = description
        self.tools = tools or []  # if no tools passed, default to empty list

    @abstractmethod
    def run(self, task: str, context: dict = None) -> str:
        """
        Execute this agent on a given task.

        Parameters
        ----------
        task : str
            The instruction or question this agent needs to handle.

        context : dict, optional
            Background information for this agent. Can contain:
            - previous_outputs: outputs from earlier pipeline stages
            - memory: relevant records retrieved from past sessions
            - metadata: any other information the agent might need
            Defaults to empty dict if nothing is passed.

        Returns
        -------
        str
            The agent's response. Always a string so Mnemo can
            log, hash, and pass it to the judge consistently.
        """
        pass  # subclasses must implement this, hence abstractmethod

    def __repr__(self):
        # makes agents print nicely in logs and debug output
        # e.g. MnemoAgent(name=researcher-llama3, tools=0)
        return f"MnemoAgent(name={self.name}, tools={len(self.tools)})"