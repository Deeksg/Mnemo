# mnemo/core.py
#
# The front door of Mnemo.
# This is the ONLY thing clients ever import.
# Everything else is hidden behind this interface.
#
# Client workflow:
#
#   from mnemo.core import MnemoCore
#
#   mnemo = MnemoCore()
#   mnemo.register_stage("research", [agent_a, agent_b, agent_c])
#   mnemo.register_stage("writing",  [agent_d, agent_e, agent_f])
#   result = mnemo.run("some task")
#
# Or for flat competition (all agents same job):
#
#   mnemo = MnemoCore()
#   mnemo.register_agent(agent_a, stage="general")
#   mnemo.register_agent(agent_b, stage="general")
#   result = mnemo.run("some task")
#
# Mnemo handles everything else:
# pipeline orchestration, judge evaluation,
# reputation tracking, memory, blockchain anchoring.
# None of that is the client's concern.

from mnemo.pipeline import MnemoPipeline, PipelineStage


class MnemoCore:
    """
    The central interface for the Mnemo system.

    Clients register their agents here and call run().
    MnemoCore handles all internal orchestration.

    Parameters
    ----------
    min_interactions : int
        How many evaluations each agent needs before
        the router switches from exploration to exploitation.
        Default 3 — low because GNN is pre-trained on
        Chatbot Arena and AgentBench datasets and already
        has strong priors about what good performance looks like.
        These 3 interactions personalise those priors to
        the specific agents in this deployment.
    """

    def __init__(self, min_interactions: int = 3):

        # min_interactions controls exploration vs exploitation.
        # Stored here and passed to MnemoPipeline when built.
        self.min_interactions = min_interactions

        # _stages is an ordered dictionary.
        # Key: stage name (string)
        # Value: list of agent objects for that stage
        #
        # Why ordered dict and not a regular dict?
        # Pipeline stages must run in registration order.
        # Research must run before writing.
        # Writing must run before fact checking.
        # Regular dicts in Python 3.7+ maintain insertion
        # order — so the order client registers stages
        # is the order they run. This is intentional.
        #
        # Starts empty — filled by register_stage()
        # and register_agent()
        self._stages = {}

        # The pipeline gets built lazily — only when run()
        # is called, not when MnemoCore is created.
        # This is because stages might be registered across
        # multiple calls before run() is ever called.
        # We can't build the pipeline until all stages
        # are registered. So we wait until run().
        self._pipeline = None

        # Tracks whether the pipeline has been built yet.
        # False until run() is called for the first time.
        self._pipeline_built = False

    def register_stage(self, name: str, agents: list) -> None:
        """
        Registers a named stage with a list of competing agents.
        Stages run in the order they are registered.

        Use this when your pipeline has multiple distinct stages
        where different agents handle different responsibilities.

        Example — our reference implementation:
            mnemo.register_stage("research",
                [ResearcherLlama(), ResearcherGemini(), ResearcherQwen()])
            mnemo.register_stage("writing",
                [WriterLlama(), WriterGemini(), WriterQwen()])
            mnemo.register_stage("fact_checking",
                [FactCheckerLlama(), FactCheckerGemini(), FactCheckerQwen()])

        Example — a client's custom pipeline:
            mnemo.register_stage("analysis",
                [their_analyst_a, their_analyst_b])
            mnemo.register_stage("review",
                [their_reviewer_a, their_reviewer_b])

        Parameters
        ----------
        name : str
            What to call this stage.
            Also becomes the context key — agents in later
            stages can read the winning output of this stage
            via context.get(name).

        agents : list
            The agents that compete in this stage.
            All must have .name (str) and .run(task, context).
        """

        # Validate that agents list is not empty.
        # An empty stage would cause the pipeline to crash
        # when trying to run zero agents in parallel.
        if not agents:
            raise ValueError(
                f"Stage '{name}' must have at least one agent. "
                f"Received empty list."
            )

        # Validate that all agents have required interface.
        # This catches client mistakes early — before run()
        # is called — with a clear error message.
        for agent in agents:
            if not hasattr(agent, "name"):
                raise ValueError(
                    f"Agent in stage '{name}' is missing .name attribute. "
                    f"All agents must have a name."
                )
            if not hasattr(agent, "run"):
                raise ValueError(
                    f"Agent '{getattr(agent, 'name', 'unknown')}' "
                    f"in stage '{name}' is missing .run() method. "
                    f"All agents must implement run(task, context)."
                )

        # Store the stage.
        # If a stage with this name already exists,
        # it gets replaced — allows updating stages
        # before run() is called.
        self._stages[name] = agents

        # Reset pipeline so it gets rebuilt on next run()
        # with the updated stage configuration.
        self._pipeline_built = False

        print(
            f"[MnemoCore] Stage '{name}' registered "
            f"with {len(agents)} agent(s): "
            f"{[a.name for a in agents]}"
        )

    def register_agent(self, agent, stage: str = "general") -> None:
        """
        Registers a single agent into a stage.
        Call multiple times to add multiple agents to a stage.

        Use this for flat competition — multiple agents
        all doing the same job, no pipeline stages needed.

        Example:
            mnemo.register_agent(agent_a, stage="general")
            mnemo.register_agent(agent_b, stage="general")
            mnemo.register_agent(agent_c, stage="general")
            result = mnemo.run("some task")
            # All three compete, judge picks best, done.

        Or mix with register_stage for custom configurations:
            mnemo.register_stage("research", [agent_a, agent_b])
            mnemo.register_agent(agent_c, stage="writing")
            mnemo.register_agent(agent_d, stage="writing")

        Parameters
        ----------
        agent : any agent with .name and .run()
            The agent to register.

        stage : str
            Which stage to add this agent to.
            Default "general" for flat competition.
            Creates the stage automatically if it
            doesn't exist yet.
        """

        # Validate agent interface
        if not hasattr(agent, "name"):
            raise ValueError(
                "Agent is missing .name attribute."
            )
        if not hasattr(agent, "run"):
            raise ValueError(
                f"Agent '{getattr(agent, 'name', 'unknown')}' "
                f"is missing .run() method."
            )

        # If this stage doesn't exist yet, create it
        # with an empty list. Then append the agent.
        # This allows calling register_agent() multiple
        # times to build up a stage incrementally.
        if stage not in self._stages:
            self._stages[stage] = []

        self._stages[stage].append(agent)

        # Reset pipeline — needs rebuilding with new agent
        self._pipeline_built = False

        print(
            f"[MnemoCore] Agent '{agent.name}' "
            f"registered in stage '{stage}'"
        )

    def _build_pipeline(self) -> None:
        """
        Builds the MnemoPipeline from registered stages.
        Called automatically by run() before first execution.

        This is where PipelineStage objects are created —
        wrapping each registered stage name and agent list
        into the structure the pipeline expects.

        Why build lazily (only when run() is called)?
        Because register_stage() and register_agent() might
        be called multiple times before run(). Building
        after every registration would be wasteful.
        Building once, right before first run(), is efficient.
        """

        # Must have at least one stage registered
        if not self._stages:
            raise RuntimeError(
                "No stages registered. Call register_stage() "
                "or register_agent() before run()."
            )

        # Convert our internal _stages dict into a list
        # of PipelineStage objects that MnemoPipeline expects.
        #
        # self._stages looks like:
        # {
        #     "research":     [ResearcherLlama, ResearcherGemini, ResearcherQwen],
        #     "writing":      [WriterLlama, WriterGemini, WriterQwen],
        #     "fact_checking":[FactCheckerLlama, FactCheckerGemini, FactCheckerQwen]
        # }
        #
        # .items() gives us (name, agents) pairs:
        # [("research", [...]), ("writing", [...]), ...]
        #
        # We wrap each pair into a PipelineStage object:
        # PipelineStage(name="research", agents=[...])
        # PipelineStage(name="writing",  agents=[...])
        # PipelineStage(name="fact_checking", agents=[...])
        #
        # This list comprehension does all of that in one line.
        pipeline_stages = [
            PipelineStage(name=stage_name, agents=agents)
            for stage_name, agents in self._stages.items()
        ]

        # Create the pipeline with our stages
        # and the min_interactions threshold
        self._pipeline = MnemoPipeline(
            stages=pipeline_stages,
            min_interactions=self.min_interactions
        )

        self._pipeline_built = True

        print(
            f"[MnemoCore] Pipeline built with "
            f"{len(pipeline_stages)} stage(s): "
            f"{list(self._stages.keys())}"
        )

    def run(self, task: str) -> dict:
        """
        Runs the complete Mnemo pipeline on a given task.
        This is the main method clients call.

        Automatically builds the pipeline on first call
        if it hasn't been built yet.

        Parameters
        ----------
        task : str
            The task or question to run through the pipeline.

        Returns
        -------
        dict — complete pipeline result:
            {
                "task": "original task",
                "stages": {
                    "research": {
                        "outputs": {agent_name: output},
                        "evaluation": {judge evaluation},
                        "winner": "agent_name",
                        "winner_output": "winning text"
                    },
                    "writing": { ...same... },
                    "fact_checking": { ...same... }
                },
                "final_output": "final result text",
                "exploration_mode": true/false,
                "interaction_counts": {agent_name: count}
            }
        """

        # Validate task is not empty
        if not task or not task.strip():
            raise ValueError(
                "Task cannot be empty. "
                "Provide a question or instruction."
            )

        # Build pipeline if not built yet
        # or if stages were updated since last build
        if not self._pipeline_built:
            self._build_pipeline()

        print(f"\n[MnemoCore] Running task: {task[:80]}...")

        # Hand the task to the pipeline
        # Pipeline handles all parallel execution,
        # judge evaluation, context passing between stages,
        # and interaction count tracking
        result = self._pipeline.run(task)

        # Add exploration mode status to result
        # so client can see whether Mnemo is still
        # exploring or has switched to routing
        result["exploration_mode"] = (
            self._pipeline._is_exploration_mode()
        )

        return result

    def get_reputation_scores(self) -> dict:
        """
        Returns current interaction counts per agent.
        Placeholder for Phase 4 — GNN reputation engine.
        In Phase 4 this will return actual GNN-computed
        reputation scores, not just interaction counts.
        For now returns interaction counts as a proxy.
        """
        if not self._pipeline_built:
            return {}
        return self._pipeline.interaction_counts.copy()

    def get_registered_stages(self) -> dict:
        """
        Returns the currently registered stages and agents.
        Useful for clients to verify their registration.

        Returns
        -------
        dict:
            {
                "stage_name": ["agent_name_1", "agent_name_2"],
                ...
            }
        """
        return {
            stage_name: [agent.name for agent in agents]
            for stage_name, agents in self._stages.items()
        }