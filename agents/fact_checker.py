# agents/fact_checker.py
# Three fact checker agents from three different companies.
# Verifies written output against research findings.
# Temperature 0.3 — lower than researcher/writer because
# fact checking needs consistency over creativity.

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.base import MnemoAgent

load_dotenv()


def _build_groq_client(model_name: str) -> ChatGroq:
    return ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def _build_gemini_client() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("AGENT_B_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )


# Fact-CheckerLlama and Fact-CheckerQwen — primary on Groq
# Fallback → Gemini (Google infrastructure)
def _build_fallback_for_groq_agents() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("AGENT_B_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )

# Fact-CheckerGemini — primary on Google
# Fallback → Llama on Groq (different infrastructure)
def _build_fallback_for_gemini() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("AGENT_A_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


FACT_CHECKER_BASE_INSTRUCTIONS = """You are a fact checking agent.
You receive a written response and verify its accuracy
against the research findings provided.

Your output must always follow this exact structure:
1. VERDICT: One of — APPROVED / APPROVED WITH NOTES / NEEDS REVISION
2. ACCURACY SCORE: A number from 0 to 10
3. QUALITY SCORE: A number from 0 to 10
4. ISSUES FOUND: List factual errors, unsupported claims,
   or logical inconsistencies. Write NONE if everything checks out.
5. SUGGESTIONS: Specific improvements if needed.
   Write NONE if the response is solid.

Be critical but fair. Your job is quality control, not rewriting."""


class FactCheckerLlama(MnemoAgent):
    """
    Fact checker powered by Meta's Llama 3.1 8B via Groq.
    Fast and decisive. Strong at quickly identifying
    clear factual errors and unsupported claims.
    """

    def __init__(self):
        super().__init__(
            name="fact-checker-llama",
            description=(
                "Meta's Llama 3.1 8B fact checker. Fast and decisive. "
                "Strong at identifying clear factual errors quickly."
            ),
            tools=[]
        )
        self.llm = _build_groq_client(
            os.getenv("AGENT_A_MODEL", "llama-3.1-8b-instant")
        )
        self.fallback_llm = _build_fallback_for_groq_agents()

    def run(self, task: str, context: dict = None) -> str:
        context = context or {}
        system_prompt = (
            FACT_CHECKER_BASE_INSTRUCTIONS
            + "\n\nPersonality: Be decisive. "
            "Flag issues clearly and quickly without hesitation."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Original topic: {task}\n\n"
                f"Research findings:\n"
                f"{context.get('research', 'No research provided.')}\n\n"
                f"Written response to fact check:\n"
                f"{context.get('writing', 'No written output provided.')}" 
                # "written_output" changed to "writing" because pipeline.py uses stage name
            ))
        ]
        if context.get("memory"):
            memory_text = "\n".join(context["memory"])
            messages.insert(1, HumanMessage(
                content=f"Relevant past context:\n{memory_text}"
            ))
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(
                f"[FactCheckerLlama] Primary failed: {e}. "
                "Switching to fallback."
            )
            response = self.fallback_llm.invoke(messages)
            return response.content


class FactCheckerGemini(MnemoAgent):
    """
    Fact checker powered by Google's Gemini 2.5 Flash.
    Thorough and balanced. Google's training on factual
    grounding makes it strong at identifying subtle
    inaccuracies and missing context.
    """

    def __init__(self):
        super().__init__(
            name="fact-checker-gemini",
            description=(
                "Google's Gemini 2.5 Flash fact checker. Thorough "
                "and balanced. Strong at subtle inaccuracies and "
                "missing context."
            ),
            tools=[]
        )
        self.llm = _build_gemini_client()
        self.fallback_llm = _build_fallback_for_gemini()

    def run(self, task: str, context: dict = None) -> str:
        context = context or {}
        system_prompt = (
            FACT_CHECKER_BASE_INSTRUCTIONS
            + "\n\nPersonality: Be thorough. "
            "Check every claim carefully and flag missing context."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Original topic: {task}\n\n"
                f"Research findings:\n"
                f"{context.get('research', 'No research provided.')}\n\n"
                f"Written response to fact check:\n"
                f"{context.get('writing', 'No written output provided.')}"
            ))
        ]
        if context.get("memory"):
            memory_text = "\n".join(context["memory"])
            messages.insert(1, HumanMessage(
                content=f"Relevant past context:\n{memory_text}"
            ))
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(
                f"[FactCheckerGemini] Primary failed: {e}. "
                "Switching to fallback."
            )
            response = self.fallback_llm.invoke(messages)
            return response.content


class FactCheckerQwen(MnemoAgent):
    """
    Fact checker powered by Alibaba's Qwen3 32B via Groq.
    Precise and technical. Qwen's STEM training makes it
    strong at identifying technical inaccuracies and
    logical inconsistencies in structured content.
    """

    def __init__(self):
        super().__init__(
            name="fact-checker-qwen",
            description=(
                "Alibaba's Qwen3 32B fact checker. Precise and "
                "technical. Strong at technical errors and "
                "logical inconsistencies."
            ),
            tools=[]
        )
        self.llm = _build_groq_client(
            os.getenv("AGENT_C_MODEL", "qwen/qwen3-32b")
        )
        self.fallback_llm = _build_fallback_for_groq_agents()

    def run(self, task: str, context: dict = None) -> str:
        context = context or {}
        system_prompt = (
            FACT_CHECKER_BASE_INSTRUCTIONS
            + "\n\nPersonality: Be precise. "
            "Surface even subtle technical inconsistencies."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Original topic: {task}\n\n"
                f"Research findings:\n"
                f"{context.get('research', 'No research provided.')}\n\n"
                f"Written response to fact check:\n"
                f"{context.get('writing', 'No written output provided.')}"
            ))
        ]
        if context.get("memory"):
            memory_text = "\n".join(context["memory"])
            messages.insert(1, HumanMessage(
                content=f"Relevant past context:\n{memory_text}"
            ))
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(
                f"[FactCheckerQwen] Primary failed: {e}. "
                "Switching to fallback."
            )
            response = self.fallback_llm.invoke(messages)
            return response.content