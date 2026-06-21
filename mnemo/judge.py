# mnemo/judge.py
# The judge agent — evaluates multiple agent outputs per stage
# and returns structured scores that feed the GNN reputation engine.
#
# Model: Llama 3.3 70B via Groq (primary)
# Fallback: Gemini 2.5 Flash via Google AI Studio
# Cross-provider fallback — two completely different infrastructures.
# If Groq's entire network fails, Google's servers are unaffected.
# Probability of both failing simultaneously during a demo: essentially zero.
#
# Why Llama 3.3 70B as judge and not a worker agent model?
# The judge must be architecturally separate from worker agents
# to prevent style bias. A judge naturally favours outputs that
# resemble its own training style. Llama 3.3 70B is a different
# generation and size class from Llama 3.1 8B (worker A),
# Gemini Flash (worker B), and Qwen3 32B (worker C).
# No worker agent uses Llama 3.3 70B — so the judge has
# no model-familiarity bias toward any specific worker.
#
# Never replaced by clients — always Mnemo's own.

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


class JudgeAgent:
    """
    Evaluates multiple agent outputs on the same task.
    Returns structured JSON scores consumed by the GNN.

    Not a MnemoAgent subclass — the judge is Mnemo Core,
    not the reference implementation.
    Clients never replace it.
    """

    def __init__(self):
        self.name = "judge"

        # Primary — Llama 3.3 70B on Groq
        self.llm = ChatGroq(
            model=os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )

        # Fallback — Gemini 2.5 Flash on Google AI Studio
        # Completely different infrastructure from Groq
        # If Groq is down, Google's servers are unaffected
        self.fallback_llm = ChatGoogleGenerativeAI(
            model=os.getenv("AGENT_B_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
        )

    def evaluate(self, task: str, stage: str, outputs: dict) -> dict:
        """
        Evaluate multiple agent outputs on the same task.

        Parameters
        ----------
        task : str
            The original task all agents received.

        stage : str
            Pipeline stage — "research", "writing", "fact_checking"

        outputs : dict
            agent_name -> agent_output text
            e.g. {
                "researcher-llama": "output...",
                "researcher-gemini": "output...",
                "researcher-qwen": "output..."
            }

        Returns
        -------
        dict:
            {
                "winner": "agent_name",
                "reasoning": "why this agent won",
                "scores": {
                    "agent_name": {
                        "quality": 0-10,
                        "relevance": 0-10,
                        "completeness": 0-10,
                        "overall": 0-10
                    }
                },
                "stage": "research",
                "used_fallback": false,
                "parse_error": false
            }
        """

        # Build a formatted block showing all agent outputs
        # side by side so the judge sees everything at once
        outputs_formatted = ""
        for agent_name, output in outputs.items():
            outputs_formatted += f"\n--- {agent_name} ---\n{output}\n"

        # The f-string at the start of system_prompt means
        # we can inject {stage} directly into the prompt text
        # so the judge knows which pipeline stage it's evaluating
        system_prompt = f"""You are an impartial judge evaluating AI agent outputs.
Multiple agents received the same task. Evaluate their outputs fairly.

Current pipeline stage: {stage}

Scoring criteria:
- quality: How well written and clear is the output? (0-10)
- relevance: How well does it address the actual task? (0-10)
- completeness: Does it cover what was needed fully? (0-10)
- overall: Your holistic score combining all criteria (0-10)

CRITICAL: Respond with ONLY valid JSON. No text before or after.
No markdown. No backticks. Raw JSON only.

Required format:
{{
    "winner": "agent_name_here",
    "reasoning": "one clear sentence explaining why this agent won",
    "scores": {{
        "agent_name": {{
            "quality": 0,
            "relevance": 0,
            "completeness": 0,
            "overall": 0
        }}
    }},
    "stage": "{stage}",
    "used_fallback": false,
    "parse_error": false
}}

Use exact agent names from the outputs provided.
All scores are integers 0-10.
Winner is the agent with highest overall score.
If scores are tied, pick the one with better quality."""

        # Build the message list exactly like we do in agents
        # SystemMessage = hidden instruction shaping judge behaviour
        # HumanMessage = the actual task and agent outputs to evaluate
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Task: {task}\n\n"
                f"Agent outputs:\n{outputs_formatted}"
            ))
        ]

        # Track whether fallback was used
        # This gets stored in the evaluation result
        # so the logger can flag low-confidence evaluations
        used_fallback = False

        # Try primary judge first
        try:
            response = self.llm.invoke(messages)
            result_text = response.content

        except Exception as e:
            # Primary (Groq) failed — could be rate limit,
            # network issue, or API error
            print(f"[Judge] Primary failed: {e}. Switching to fallback.")
            used_fallback = True

            # Try fallback judge (Google)
            try:
                response = self.fallback_llm.invoke(messages)
                result_text = response.content

            except Exception as e2:
                # Both providers failed simultaneously
                # Return neutral scores so pipeline doesn't crash
                print(f"[Judge] Fallback also failed: {e2}.")
                return self._neutral_scores(
                    outputs, stage, used_fallback=True
                )

        # Try to parse the JSON response from the judge
        # json.loads() converts the raw JSON string into
        # a Python dictionary we can work with
        try:
            result = json.loads(result_text)
            result["used_fallback"] = used_fallback
            result["parse_error"] = False
            return result

        except json.JSONDecodeError:
            # Model returned text instead of valid JSON
            # despite explicit instructions not to
            # Return neutral scores rather than crashing
            print("[Judge] JSON parse failed. Returning neutral scores.")
            return self._neutral_scores(
                outputs, stage,
                used_fallback=used_fallback,
                parse_error=True
            )

    def _neutral_scores(
        self,
        outputs: dict,
        stage: str,
        used_fallback: bool = False,
        parse_error: bool = False
    ) -> dict:
        """
        Returns neutral 5/10 scores for all agents.
        Used when both primary and fallback fail completely,
        or when JSON parsing fails after successful API call.
        5/10 is deliberately neutral — not high enough to
        unfairly boost any agent's reputation, not low enough
        to unfairly damage it. The GNN weights evaluations
        with parse_error=True as low confidence.
        """
        return {
            "winner": list(outputs.keys())[0],
            "reasoning": "Neutral fallback — evaluation failed",
            "scores": {
                name: {
                    "quality": 5,
                    "relevance": 5,
                    "completeness": 5,
                    "overall": 5
                } for name in outputs.keys()
            },
            "stage": stage,
            "used_fallback": used_fallback,
            "parse_error": parse_error
        }

    def get_winner_output(
        self,
        evaluation: dict,
        outputs: dict
    ) -> str:
        """
        Returns the actual text output of the winning agent.
        Called by the pipeline after each stage evaluation
        to get the content to pass to the next stage.
        Example: after research stage, the winning researcher's
        output gets passed as context['research'] to the writers.
        """
        winner_name = evaluation["winner"]
        # .get() with fallback to first value protects against
        # the rare case where winner name in evaluation doesn't
        # match any key in outputs dict
        return outputs.get(winner_name, list(outputs.values())[0])