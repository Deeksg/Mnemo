# mnemo/pipeline.py
#
# Dynamic pipeline — works with any number of stages,
# any stage names, any agents.
# No hardcoded roles, no hardcoded stage names.
#
# HOW IT WORKS:
# The pipeline holds an ordered list of PipelineStage objects.
# Each PipelineStage has a name and a list of agents.
# When run() is called, the pipeline loops through every stage
# in order, runs all agents in that stage simultaneously,
# passes outputs to the judge, stores the winner's output
# in context under the stage name, and passes that context
# to the next stage.
#
# This means:
# - A client with 2 stages works fine
# - A client with 10 stages works fine
# - Stage names can be anything
# - Any number of agents per stage
# - Our reference implementation is just one example
#
# EXPLORATION VS EXPLOITATION:
# The pipeline tracks how many interactions each agent has had.
# While any agent has fewer than min_interactions evaluations,
# ALL agents compete on every task (exploration mode).
# Once all agents have enough data, the router takes over
# and only the best agent runs (exploitation mode).
# With a pre-trained GNN, min_interactions is low (3)
# because the GNN already has strong priors from training data.

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from mnemo.judge import JudgeAgent


@dataclass
class PipelineStage:
    """
    Represents one stage in the pipeline.
    Just a name and a list of agents that compete in that stage.

    name : str
        What this stage is called.
        Used as the context key for passing output to next stage.
        Example: "research", "writing", "fact_checking"
        Or for a client: "analysis", "review", "approval"

    agents : list
        The agents that compete in this stage.
        All receive the same task and context simultaneously.
        Judge picks the best output after all finish.
    """
    name: str
    agents: list = field(default_factory=list)


class MnemoPipeline:
    """
    Runs a dynamic multi-stage competition pipeline.

    Works with any number of stages and any agent configuration.
    Agents are passed in — never imported or hardcoded here.

    Parameters
    ----------
    stages : list of PipelineStage
        The ordered list of stages to run.
        Created by MnemoCore from registered agents.

    min_interactions : int
        How many evaluations each agent needs before
        the router can take over from full competition.
        Default 3 — low because GNN is pre-trained.
    """

    def __init__(
        self,
        stages: list,
        min_interactions: int = 3,
        logger=None  # NEW — optional logger to load counts from

    ):
        # The ordered list of stages
        # Each is a PipelineStage with name and agents
        self.stages = stages

        # Track how many times each agent has competed.
        # If a logger is provided, load counts from the database
        # so they persist across separate program runs.
        # Without this, counts would reset to 0 every time
        # main.py is run, meaning exploration mode never ends.
        self.interaction_counts = {}
        for stage in self.stages:
            for agent in stage.agents:
                if logger:
                    self.interaction_counts[agent.name] = (
                        logger.get_agent_appearance_count(agent.name)
                    )
                else:
                    # No logger — start from zero (useful for testing)
                    self.interaction_counts[agent.name] = 0

        # Single judge shared across all stages and all runs
        # Consistent evaluator = comparable reputation scores
        self.judge = JudgeAgent()

        # Thread pool for parallel agent execution
        # max_workers = largest number of agents in any single stage
        # so we always have enough threads for the biggest stage
        max_agents_per_stage = max(
            len(stage.agents) for stage in self.stages
        )
        self.executor = ThreadPoolExecutor(
            max_workers=max_agents_per_stage
        )

    def _is_exploration_mode(self) -> bool:
        """
        Returns True if ANY agent still needs more interactions
        before the router can make reliable decisions.

        As long as even one agent is under the threshold,
        we keep all agents competing so nobody gets left
        without enough evaluation data.

        With a pre-trained GNN, this threshold is low (3)
        because the model already has strong priors from
        Chatbot Arena and AgentBench training data.
        These 3 interactions just personalise the pre-trained
        model to the specific agents in this deployment.
        """
        return any(
            count < self.min_interactions
            for count in self.interaction_counts.values()
        )

    def _update_interaction_counts(self, evaluation: dict):
        """
        After each stage evaluation, increments the interaction
        count for every agent that was evaluated in that stage.
        Once all agents hit min_interactions, exploration ends.
        """
        for agent_name in evaluation.get("scores", {}).keys():
            if agent_name in self.interaction_counts:
                self.interaction_counts[agent_name] += 1

    def _run_agent(
        self,
        agent,
        task: str,
        context: dict
    ) -> tuple:
        """
        Runs one agent and returns (name, output) as a tuple.

        WHY A TUPLE?
        When three agents run in parallel we need to keep
        each output linked to the agent that produced it.
        Returning (name, output) together means we never
        lose track of who said what.

        dict([("agent-a", "text"), ("agent-b", "text")])
        becomes {"agent-a": "text", "agent-b": "text"}
        which is exactly what judge.evaluate() expects.

        agent.run() is defined in each agent file:
        agents/researcher.py → ResearcherLlama.run()
        agents/writer.py     → WriterLlama.run()
        etc.
        Or in a client's own agent files — doesn't matter.
        As long as run() exists and returns a string, it works.
        """
        output = agent.run(task, context)
        return (agent.name, output)

    async def _run_stage_parallel(
        self,
        stage: PipelineStage,
        task: str,
        context: dict
    ) -> dict:
        """
        Runs all agents in one stage simultaneously using threads.
        Returns dict of agent_name -> output_text.

        WHY THREADS FOR PARALLEL EXECUTION?
        LangChain's .invoke() is a blocking call — when it runs,
        your code completely freezes and waits for the API response.
        You cannot just use async/await on it directly.

        Solution: run each blocking call in its own thread.
        Threads are independent — one thread freezing waiting
        for an API response doesn't stop other threads from running.

        asyncio coordinates the threads:
        - run_in_executor hands work to threads (non-blocking)
        - asyncio.gather waits for ALL threads to finish
        - Results are collected and returned together

        Sequential (old way):   3 agents × 3 seconds = 9 seconds
        Parallel (this way):    3 agents running together = 3 seconds
        """

        # Get reference to the running event loop
        # The event loop is the coordinator — it manages
        # all async tasks and knows when threads finish.
        # get_event_loop() doesn't create a new one —
        # it gets the one already running right now.
        # We need this reference to call run_in_executor.
        loop = asyncio.get_event_loop()

        # Hand each agent to the thread pool simultaneously.
        # run_in_executor does NOT wait — it submits work
        # to a thread and immediately returns a Future
        # (a promise that the result will arrive eventually).
        # All three submissions happen before any agent finishes.
        # This is the moment all agents start running in parallel.
        #
        # Arguments:
        # self.executor    — which thread pool to use
        # self._run_agent  — the function to run in the thread
        #                    (notice: no brackets, passing the
        #                    function itself not calling it)
        # agent, task, context — arguments for _run_agent
        futures = [
            loop.run_in_executor(
                self.executor,
                self._run_agent,
                agent,
                task,
                context
            )
            for agent in stage.agents
        ]

        # Wait for ALL agents to finish before continuing.
        # asyncio.gather takes multiple futures and waits
        # until every single one completes.
        # *futures unpacks the list — gather needs individual
        # items not a list: gather(a, b, c) not gather([a,b,c])
        #
        # results is a list of tuples:
        # [("researcher-llama", "output text"),
        #  ("researcher-gemini", "output text"),
        #  ("researcher-qwen", "output text")]
        results = await asyncio.gather(*futures)

        # dict() converts list of (key, value) tuples
        # into a proper dictionary:
        # {"researcher-llama": "output text", ...}
        # This is exactly what judge.evaluate() expects.
        return dict(results)

    def run(self, task: str) -> dict:
        """
        Public entry point — runs the complete pipeline.
        Called by MnemoCore.run() after agents are registered.
        Can also be called directly for testing.

        asyncio.run() starts a fresh event loop, runs
        _run_async to completion, closes the loop,
        and returns the result.

        This gives callers a clean synchronous interface —
        they just call pipeline.run("task") like any
        normal function without knowing about async.
        """
        return asyncio.run(self._run_async(task))

    async def _run_async(self, task: str) -> dict:
        """
        The actual pipeline execution.
        Loops through every registered stage in order.
        At each stage: runs agents in parallel, judge evaluates,
        winner's output stored in context for next stage.

        Separated from run() because asyncio.run() needs
        to call an async function. run() is synchronous
        so it calls asyncio.run() which calls this.
        """

        # Result container — filled as we go through stages
        result = {
            "task": task,
            "stages": {},
            "final_output": None,
            "exploration_mode": self._is_exploration_mode()
        }

        # Context grows as stages complete.
        # After stage named "research":
        #   context = {"research": "winning output text"}
        # After stage named "writing":
        #   context = {"research": "...", "writing": "..."}
        # Each stage's winner output is stored under the stage name.
        # Agents in later stages read whatever context keys they need.
        context = {}

        # Loop through every stage in registration order
        # This is the key change from the old hardcoded pipeline —
        # we don't know or care how many stages there are
        # or what they're called. We just run whatever was registered.
        for stage in self.stages:

            print(f"\n[Mnemo] Stage '{stage.name}' starting...")

            # Run all agents in this stage simultaneously
            stage_outputs = await self._run_stage_parallel(
                stage, task, context
            )

            # Judge evaluates all outputs for this stage
            # judge.evaluate() is in mnemo/judge.py
            # Takes: task, stage name, dict of agent outputs
            # Returns: winner name, reasoning, scores per agent
            evaluation = self.judge.evaluate(
                task=task,
                stage=stage.name,
                outputs=stage_outputs
            )

            # Update interaction counts for all agents
            # that were evaluated in this stage.
            # Once all agents hit min_interactions,
            # _is_exploration_mode() returns False
            # and the router takes over.
            self._update_interaction_counts(evaluation)

            # Get the actual text the winning agent produced.
            # judge.get_winner_output() is in mnemo/judge.py
            # It looks up the winner's name in stage_outputs
            # and returns their text.
            winning_output = self.judge.get_winner_output(
                evaluation, stage_outputs
            )

            # Store complete stage record
            # Logger in Phase 3 reads this entire structure
            result["stages"][stage.name] = {
                "outputs": stage_outputs,
                "evaluation": evaluation,
                "winner": evaluation["winner"],
                "winner_output": winning_output
            }

            # THE KEY HANDOFF BETWEEN STAGES
            # Store winning output under this stage's name.
            # Next stage's agents read context.get(stage.name)
            # to access the previous stage's winning output.
            #
            # Our reference agents use:
            # context.get("research")      → winning researcher output
            # context.get("written_output") → winning writer output
            # context.get("fact_checking") → winning fact check
            #
            # A client's agents use whatever key names
            # match their stage names.
            context[stage.name] = winning_output

            print(
                f"[Mnemo] Stage '{stage.name}' complete. "
                f"Winner: {evaluation['winner']} | "
                f"Exploration mode: {self._is_exploration_mode()}"
            )

        # Final output is the winner of the last stage
        last_stage_name = self.stages[-1].name
        result["final_output"] = (
            result["stages"][last_stage_name]["winner_output"]
        )

        # Current interaction counts — useful for
        # monitoring when exploration mode will end
        result["interaction_counts"] = (
            self.interaction_counts.copy()
        )

        print("\n[Mnemo] Pipeline complete.")
        return result