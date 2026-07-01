# main.py
#
# Entry point for Mnemo.
# This file serves two purposes:
#
# 1. REFERENCE IMPLEMENTATION
#    Shows exactly how a client integrates their agents with Mnemo.
#    Our reference agents (Llama, Gemini, Qwen) are registered
#    exactly the same way any client would register their own agents.
#
# 2. FIRST END-TO-END TEST
#    Running this file executes the complete pipeline for the
#    first time — all three stages, parallel agent execution,
#    judge evaluation, context passing between stages.
#
# To run:
#    python main.py
#
# Make sure your .env file has:
#    GROQ_API_KEY=your_key
#    GOOGLE_API_KEY=your_key

import json
from mnemo.core import MnemoCore
from agents.researcher import ResearcherLlama, ResearcherGemini, ResearcherQwen
from agents.writer import WriterLlama, WriterGemini, WriterQwen
from agents.fact_checker import FactCheckerLlama, FactCheckerGemini, FactCheckerQwen
from mnemo.logger import InteractionLogger


def print_separator(title: str) -> None:
    """
    Prints a clean visual separator with a title.
    Used to make terminal output readable when
    results from multiple stages print at once.
    Just a cosmetic helper — no logic inside.
    """
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_stage_results(stage_name: str, stage_data: dict) -> None:
    """
    Prints the results of one pipeline stage in a readable format.

    Parameters
    ----------
    stage_name : str
        The name of the stage — "research", "writing", "fact_checking"

    stage_data : dict
        The data for this stage from the pipeline result.
        Contains outputs, evaluation, winner, winner_output.
        This comes directly from result["stages"][stage_name]
        which the pipeline built and returned.
    """
    print_separator(f"STAGE: {stage_name.upper()}")

    # Print each agent's output
    # stage_data["outputs"] is a dict of agent_name -> output_text
    # .items() gives us both the name and the text together
    print("\n── Agent Outputs ──")
    for agent_name, output in stage_data["outputs"].items():
        print(f"\n[{agent_name}]")
        # [:500] takes only first 500 characters
        # Full outputs can be thousands of words
        # We truncate for readability in the terminal
        # The full output is still in stage_data["outputs"]
        print(output[:500] + "..." if len(output) > 500 else output)

    # Print judge evaluation
    # stage_data["evaluation"] is the dict returned by judge.evaluate()
    # Contains winner, reasoning, scores, stage, used_fallback
    print("\n── Judge Evaluation ──")
    evaluation = stage_data["evaluation"]
    print(f"Winner:    {evaluation['winner']}")
    print(f"Reasoning: {evaluation['reasoning']}")

    # Print scores for each agent
    # evaluation["scores"] is a dict of agent_name -> score_dict
    # Each score_dict has quality, relevance, completeness, overall
    print("\nScores:")
    for agent_name, scores in evaluation["scores"].items():
        print(
            f"  {agent_name}: "
            f"quality={scores['quality']} | "
            f"relevance={scores['relevance']} | "
            f"completeness={scores['completeness']} | "
            f"overall={scores['overall']}"
        )

    # Warn if judge used fallback or had parse error
    # These flags were set in mnemo/judge.py
    # used_fallback means Groq failed and Google was used
    # parse_error means JSON parsing failed and neutral scores used
    if evaluation.get("used_fallback"):
        print("\n⚠️  Judge used fallback provider for this evaluation")
    if evaluation.get("parse_error"):
        print("\n⚠️  Judge parse error — neutral scores used")

    # Print winning output in full
    print(f"\n── Winner Output ({stage_data['winner']}) ──")
    print(stage_data["winner_output"])


def run_mnemo(task: str) -> dict:
    """
    Sets up and runs the complete Mnemo pipeline.

    This function is intentionally separated from main()
    so it can be imported and called from other files
    during testing without running the full main() block.

    Parameters
    ----------
    task : str
        The task to run through the pipeline.

    Returns
    -------
    dict
        Complete pipeline result from MnemoCore.run()
    """
    # Logger created first so it can be passed into MnemoCore
    # MnemoCore passes it to MnemoPipeline which uses it to
    # load persistent interaction counts from the database
    logger = InteractionLogger()
    mnemo = MnemoCore(min_interactions=3, logger=logger)

    # ── STEP 1: Create MnemoCore ──────────────────────────
    # MnemoCore is the front door of Mnemo.
    # min_interactions=3 means exploration mode lasts
    # until every agent has been evaluated 3 times.
    # After that the router takes over.
    # Low threshold because GNN is pre-trained.
    # Logger created first so it can be passed into MnemoCore
    # MnemoCore passes it to MnemoPipeline which uses it to
    # load persistent interaction counts from the database
    # Create the logger — connects to (or creates) mnemo.db
    logger = InteractionLogger()
    mnemo = MnemoCore(min_interactions=3, logger=logger)    

    
    # ── STEP 2: Register Stages ───────────────────────────
    # This is exactly what a client would do with their agents.
    # We use our reference agents as the example.
    #
    # register_stage(name, agents) takes:
    # - name: what to call this stage (string)
    # - agents: list of agent objects for this stage
    #
    # Stages run in registration order:
    # research → writing → fact_checking
    #
    # The stage name becomes the context key —
    # winner of "research" stage gets stored as
    # context["research"] for the writing stage.

    mnemo.register_stage(
        "research",
        [
            ResearcherLlama(),   # Meta's Llama 3.1 8B via Groq
            ResearcherGemini(),  # Google's Gemini 2.5 Flash
            ResearcherQwen(),    # Alibaba's Qwen3 32B via Groq
        ]
    )

    mnemo.register_stage(
        "writing",
        [
            WriterLlama(),       # Meta's Llama 3.1 8B via Groq
            WriterGemini(),      # Google's Gemini 2.5 Flash
            WriterQwen(),        # Alibaba's Qwen3 32B via Groq
        ]
    )

    mnemo.register_stage(
        "fact_checking",
        [
            FactCheckerLlama(),  # Meta's Llama 3.1 8B via Groq
            FactCheckerGemini(), # Google's Gemini 2.5 Flash
            FactCheckerQwen(),   # Alibaba's Qwen3 32B via Groq
        ]
    )

    # Show what was registered — calls get_registered_stages()
    # in mnemo/core.py which returns {stage_name: [agent_names]}
    print_separator("MNEMO INITIALISED")
    print("\nRegistered stages:")
    for stage_name, agent_names in mnemo.get_registered_stages().items():
        print(f"  {stage_name}: {agent_names}")

    # ── STEP 3: Run the Pipeline ──────────────────────────
    # mnemo.run(task) triggers the entire pipeline:
    # 1. _build_pipeline() called — creates MnemoPipeline
    # 2. MnemoPipeline.run(task) called
    # 3. asyncio starts, _run_async() executes
    # 4. Each stage runs in order
    # 5. Within each stage agents run in parallel
    # 6. Judge evaluates after each stage
    # 7. Winner output passed as context to next stage
    # 8. Complete result returned
    print_separator("RUNNING PIPELINE")
    print(f"\nTask: {task}\n")

    result = mnemo.run(task)

    # Permanently save this interaction to the database.
    # Returns the id SQLite assigned to this row.
    interaction_id = logger.log_interaction(result)
    result["_db_id"] = interaction_id

    # Print reputation standings after every run
    print_reputation_summary(logger)


    return result


def print_reputation_summary(logger) -> None:
    """
    Prints current reputation standings after each run.
    Shows win rates and average scores per agent so you
    can visually see reputation building over time.
    """
    print_separator("REPUTATION STANDINGS")

    agents = [
        "researcher-llama", "researcher-gemini", "researcher-qwen",
        "writer-llama", "writer-gemini", "writer-qwen",
        "fact-checker-llama", "fact-checker-gemini", "fact-checker-qwen"
    ]

    # Group by stage for readability
    stages = {
        "Research": ["researcher-llama", "researcher-gemini", "researcher-qwen"],
        "Writing": ["writer-llama", "writer-gemini", "writer-qwen"],
        "Fact Checking": ["fact-checker-llama", "fact-checker-gemini", "fact-checker-qwen"]
    }

    for stage_label, stage_agents in stages.items():
        print(f"\n── {stage_label} ──")
        for agent_name in stage_agents:
            scores = logger.get_agent_average_scores(agent_name)
            if scores["total_appearances"] == 0:
                print(f"  {agent_name}: no data yet")
            else:
                print(
                    f"  {agent_name}: "
                    f"win rate {scores['win_rate']}% | "
                    f"avg overall {scores['avg_overall']} | "
                    f"appearances {scores['total_appearances']}"
                )

def main():
    """
    Main function — runs Mnemo on a sample task and
    prints all results in a readable format.

    The task we chose — "What are the implications of
    quantum computing on modern cryptography?" — is good
    for a first run because:
    - It requires genuine research (tests researcher agents)
    - It requires clear explanation (tests writer agents)
    - It has verifiable facts (tests fact checker agents)
    - It's technical enough to show model differences
    """

    print_separator("MNEMO — MULTI AGENT INTELLIGENCE SYSTEM")
    print("\nWelcome to Mnemo.")
    print("Three AI agents from Meta, Google, and Alibaba will")
    print("compete on your task. The best output wins each stage.\n")

    task = input("Enter your task: ").strip()

    if not task:
        print("\n❌ Task cannot be empty. Please provide a question or instruction.")
        exit()

    print(f"\n✅ Task received: {task}")

    # Run the pipeline
    # run_mnemo() returns the complete result dict
    result = run_mnemo(task)

    # ── PRINT STAGE RESULTS ───────────────────────────────
    # result["stages"] is a dict of stage_name -> stage_data
    # .items() gives us both the name and data together
    # We loop through and print each stage
    for stage_name, stage_data in result["stages"].items():
        print_stage_results(stage_name, stage_data)

    # ── PRINT FINAL SUMMARY ───────────────────────────────
    print_separator("FINAL SUMMARY")

    print(f"\nTask: {result['task']}")
    print(f"\nExploration mode: {result['exploration_mode']}")
    print(
        "(All agents competing — not enough interactions "
        "yet for router to take over)"
        if result['exploration_mode']
        else "(Router active — best agents being selected automatically)"
    )

    # Print interaction counts
    # result["interaction_counts"] is {agent_name: count}
    # Set by _update_interaction_counts() in pipeline.py
    # after each stage evaluation
    print("\nInteraction counts after this run:")
    for agent_name, count in result["interaction_counts"].items():
        print(f"  {agent_name}: {count} evaluation(s)")

    # Print stage winners summary
    print("\nStage winners:")
    for stage_name, stage_data in result["stages"].items():
        print(f"  {stage_name}: {stage_data['winner']}")

    # Print final output
    print_separator("FINAL OUTPUT")
    print(result["final_output"])

    # Save full result to JSON file for inspection
    # json.dumps() converts the result dict to a
    # formatted JSON string. indent=2 makes it readable.
    # This is useful for debugging — you can open
    # result.json and see everything in full detail.
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Full result saved to result.json")


# This block only runs when you execute main.py directly:
#   python main.py
#
# It does NOT run when main.py is imported by another file.
# This is a Python convention — if __name__ == "__main__"
# is True only when the file is the entry point.
# When imported, __name__ is the module name not "__main__".
# This protects the main() call from running accidentally
# during imports or testing.
if __name__ == "__main__":
    main()