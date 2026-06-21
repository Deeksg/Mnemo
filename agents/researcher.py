# agents/researcher.py
# Three researcher agents from three different companies.
# Meta (Llama 3.1 8B), Google (Gemini 2.5 Flash), Alibaba (Qwen3 32B).
# Same core instructions and output structure across all three.
# Only a single personality directive differs per model.
# This ensures judge scores reflect model capability, not prompt differences.

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
        temperature=0.7,
    )


def _build_gemini_client() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("AGENT_B_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )


# ResearcherLlama and ResearcherQwen — primary on Groq
# Fallback → Gemini (Google infrastructure)
def _build_fallback_for_groq_agents() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("AGENT_B_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )

# ResearcherGemini — primary on Google
# Fallback → Llama on Groq (different infrastructure)
def _build_fallback_for_gemini() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("AGENT_A_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
    )


# Core instructions identical across all three agents.
# Critical for experimental validity — if instructions differ,
# the judge might pick a winner based on prompt differences
# rather than genuine model capability.
RESEARCHER_BASE_INSTRUCTIONS = """You are a research agent.
Your job is to investigate the given topic and return structured findings.

Your output must always follow this exact structure:
1. KEY FACTS: The most important factual points
2. BACKGROUND: Relevant context and background information
3. KEY CONSIDERATIONS: Important factors to keep in mind
4. GAPS: What is uncertain or needs further investigation

Be factual and structured. Flag anything you are uncertain about clearly."""


class ResearcherLlama(MnemoAgent):
    """
    Researcher powered by Meta's Llama 3.1 8B via Groq.
    Trained on broad English web data optimized for instruction
    following. Tends toward concise, direct responses with
    strong factual grounding. Fastest of the three agents —
    sub-200ms responses on Groq's LPU hardware.
    """

    def __init__(self):
        super().__init__(
            name="researcher-llama",
            description=(
                "Meta's Llama 3.1 8B researcher. Broad English web "
                "training, strong instruction following. Fast and concise."
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
            RESEARCHER_BASE_INSTRUCTIONS
            + "\n\nPersonality: Be concise and direct. "
            "Every word must earn its place."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Research the following:\n\n{task}"
            )
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
                f"[ResearcherLlama] Primary failed: {e}. "
                "Switching to fallback."
            )
            response = self.fallback_llm.invoke(messages)
            return response.content


class ResearcherGemini(MnemoAgent):
    """
    Researcher powered by Google's Gemini 2.5 Flash
    via Google AI Studio.
    Trained on Google's multimodal data infrastructure
    with emphasis on factual grounding and structured
    comprehensiveness. Google's knowledge graph and
    search data shapes how it reasons about facts —
    completely different training pipeline from Meta
    or Alibaba. 1,500 requests/day free, backed by
    Google's infrastructure.
    """

    def __init__(self):
        super().__init__(
            name="researcher-gemini",
            description=(
                "Google's Gemini 2.5 Flash researcher. Multimodal "
                "data infrastructure training. Comprehensive and "
                "well-structured with strong factual grounding."
            ),
            tools=[]
        )
        self.llm = _build_gemini_client()
        self.fallback_llm = _build_fallback_for_gemini()

    def run(self, task: str, context: dict = None) -> str:
        context = context or {}
        system_prompt = (
            RESEARCHER_BASE_INSTRUCTIONS
            + "\n\nPersonality: Be comprehensive. Cover multiple "
            "angles and surface connections between ideas."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Research the following:\n\n{task}"
            )
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
                f"[ResearcherGemini] Primary failed: {e}. "
                "Switching to fallback."
            )
            response = self.fallback_llm.invoke(messages)
            return response.content


class ResearcherQwen(MnemoAgent):
    """
    Researcher powered by Alibaba's Qwen3 32B via Groq.
    Trained with heavy STEM and multilingual focus using
    Alibaba's data infrastructure. Strong at technical
    topics, mathematical reasoning, and structured analysis.
    Offers a genuinely different knowledge perspective —
    different cultural and academic training emphasis
    from both Meta and Google.
    """

    def __init__(self):
        super().__init__(
            name="researcher-qwen",
            description=(
                "Alibaba's Qwen3 32B researcher. STEM-focused "
                "multilingual training. Strong technical reasoning "
                "and structured depth."
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
            RESEARCHER_BASE_INSTRUCTIONS
            + "\n\nPersonality: Be detailed and precise. "
            "Surface technical depth and non-obvious insights."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Research the following:\n\n{task}"
            )
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
                f"[ResearcherQwen] Primary failed: {e}. "
                "Switching to fallback."
            )
            response = self.fallback_llm.invoke(messages)
            return response.content