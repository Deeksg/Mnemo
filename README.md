# Mnemo 🧠

> Reputation-ranked persistent memory middleware for multi-agent LLM systems

Mnemo is an open-source middleware layer that brings two critical missing primitives to multi-agent AI systems:

- **Reputation ranking** — agents are scored on real performance, not assumed to be equal
- **Verifiable memory** — every interaction is logged, hashed, and anchored on-chain so nothing can be tampered with

Drop Mnemo into any multi-agent system and it will automatically track who performs best, remember everything agents have done across sessions, and route future tasks to the highest-reputation agent for that task type.

---

## The Problem

Every major multi-agent framework today (LangGraph, CrewAI, AutoGen) treats all agents as equal. There is no memory of which agent performed well last time, no reputation built from past interactions, and no verifiable record of what any agent did or said. Tasks are assigned arbitrarily. Bad agents keep getting assigned work. Good agents are indistinguishable from bad ones.

Mnemo fixes this.

---

## How It Works

### 1. Multi-Agent Competition Pipeline

Every task runs through a three-stage pipeline. At each stage, multiple agents compete on the same task independently. A judge evaluates all outputs and picks the best one. Reputation scores update after every stage.

One full task run generates 9 reputation data points across 3 agents and 3 stages. Over hundreds of tasks, a clear picture of each agent's strengths emerges.

### 2. Reputation Engine (GNN)

Agent interactions are modelled as a graph — each agent is a node, each interaction is a weighted edge. A Graph Neural Network trained on the HuggingFace Chatbot Arena and AgentBench datasets learns what good performance looks like from millions of real human preference judgments.

Reputation scores are continuous, dynamic, and task-type-aware — an agent can have high reputation for research tasks and low reputation for fact-checking.

### 3. Verifiable Memory Layer

Every interaction produces a complete record — the task given, every agent's full output, the judge's evaluation and reasoning, quality scores per agent, the router's assignment decision, and reputation state at that moment.

These records are stored as vector embeddings in Weaviate, enabling semantic retrieval across sessions. Agents remember what they have done before.

### 4. Tamper-Proof Audit Trail

All interaction records are hashed into a Merkle tree. The root hash is anchored on the Ethereum Sepolia testnet after every session. Any modification to any past record produces a completely different root hash, making tampering immediately detectable.

### 5. Reputation-Based Routing

The router reads live reputation scores and assigns incoming tasks to the highest-ranked agent for that task type. As agents accumulate history, routing becomes increasingly intelligent and evidence-based.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM Providers | Groq (Llama 3, Mistral, Gemma 2) |
| Reputation Engine | PyTorch Geometric (GNN) |
| Training Data | Chatbot Arena + AgentBench |
| Vector Memory | Weaviate |
| Caching | Redis |
| Blockchain | Ethereum Sepolia testnet |
| Smart Contracts | Solidity |

---

## Project Structure

    mnemo/
    ├── agents/         Reference agent definitions
    ├── data/           Dataset loaders and processing
    ├── graph/          GNN reputation engine
    ├── memory/         Weaviate vector memory layer
    ├── contracts/      Solidity smart contracts
    ├── scripts/        Utility scripts
    ├── notebooks/      Google Colab training notebooks
    ├── main.py         Entry point
    └── README.md

---

## Plugin Interface

Mnemo ships with reference agents but is designed to accept any agents a client brings. Any LangGraph, CrewAI, or AutoGen agent works as a drop-in. Mnemo handles reputation tracking, memory, and routing without any changes to the agent's internal logic.

---

## Setup

Run the following in your terminal:

    git clone https://github.com/yourusername/mnemo.git
    cd mnemo
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Then add your API keys to a file named `.env` in the root folder:

    GROQ_API_KEY=your_key_here

---

## Status

🚧 Active development — Phase 2 (Multi-Agent Pipeline)

---

## License

MIT