# agents/writer.py
# Three writer agents from three different companies.
# Takes research output and produces well-structured written response.
# Same core instructions across all three for fair comparison.

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


# WriterLlama and WriterQwen — primary on Groq
# Fallback → Gemini (Google infrastructure)
def _build_fallback_for_groq_agents() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("AGENT_B_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )

# WriterGemini — primary on Google
# Fallback → Llama on Groq (different infrastructure)
def _build_fallback_for_gemini() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("AGENT_A_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
    )

WRITER_BASE_INSTRUCTIONS = """You are a writing agent.
You receive a topic and research findings and produce a written response.

Your output must always:
- Open with a strong clear summary of the topic
- Be logically structured with smooth transitions
- Use the research findings as your factual foundation
- End with a clear conclusion or key takeaway

Do not repeat the research findings verbatim.
Transform them into fluid, readable prose.
Be accurate and well structured."""


class WriterLlama(MnemoAgent):
    """
    Writer powered by Meta's Llama 3.1 8B via Groq.
    Concise, direct prose. Strong at clear communication
    without unnecessary padding.
    """

    def __init__(self):
        super().__init__(
            name="writer-llama",
            description=(
                "Meta's Llama 3.1 8B writer. Produces concise, "
                "direct prose with strong factual grounding."
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
            WRITER_BASE_INSTRUCTIONS
            + "\n\nPersonality: Be concise. "
            "Cut anything that does not add value."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Topic: {task}\n\nResearch findings:\n"
                f"{context.get('research', 'No research provided.')}"
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
                f"[WriterLlama] Primary failed: {e}. "
                "Switching to fallback."
            )
            response = self.fallback_llm.invoke(messages)
            return response.content


class WriterGemini(MnemoAgent):
    """
    Writer powered by Google's Gemini 2.5 Flash.
    Comprehensive, well-structured prose. Strong at
    covering multiple angles with balanced coverage.
    """

    def __init__(self):
        super().__init__(
            name="writer-gemini",
            description=(
                "Google's Gemini 2.5 Flash writer. Comprehensive, "
                "balanced prose with strong structural organization."
            ),
            tools=[]
        )
        self.llm = _build_gemini_client()
        self.fallback_llm = _build_fallback_for_gemini()

    def run(self, task: str, context: dict = None) -> str:
        context = context or {}
        system_prompt = (
            WRITER_BASE_INSTRUCTIONS
            + "\n\nPersonality: Be comprehensive. "
            "Cover all angles fairly and with balance."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Topic: {task}\n\nResearch findings:\n"
                f"{context.get('research', 'No research provided.')}"
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
                f"[WriterGemini] Primary failed: {e}. "
                "Switching to fallback."
            )
            response = self.fallback_llm.invoke(messages)
            return response.content


class WriterQwen(MnemoAgent):
    """
    Writer powered by Alibaba's Qwen3 32B via Groq.
    Detailed, technically precise prose. Strong at
    depth and structured technical writing.
    """

    def __init__(self):
        super().__init__(
            name="writer-qwen",
            description=(
                "Alibaba's Qwen3 32B writer. Detailed, technically "
                "precise prose with strong depth and structure."
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
            WRITER_BASE_INSTRUCTIONS
            + "\n\nPersonality: Be detailed. "
            "Go beyond the obvious and surface deeper insights."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Topic: {task}\n\nResearch findings:\n"
                f"{context.get('research', 'No research provided.')}"
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
                f"[WriterQwen] Primary failed: {e}. "
                "Switching to fallback."
            )
            response = self.fallback_llm.invoke(messages)
            return response.content