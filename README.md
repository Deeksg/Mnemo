# Mnemo 🧠

> Reputation-ranked persistent memory for multi-agent LLM systems

Mnemo is an open-source middleware layer that gives multi-agent AI systems two things they currently lack: **persistent memory across sessions** and **reputation-based task routing** — so the best agent for a job is always chosen based on verified past performance, not random assignment.

---

## How It Works

1. Agents complete tasks and interact with each other
2. Every interaction is logged and fed into a Graph Neural Network (GNN)
3. The GNN scores each agent's reputation based on real performance data
4. A Merkle tree hashes the memory state for tamper-proof verification
5. The memory root hash is anchored on the Ethereum Sepolia testnet
6. New tasks are automatically routed to the highest-reputation agent

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| Reputation Engine | PyTorch Geometric (GNN) |
| Vector Memory | Weaviate |
| Training Data | HuggingFace Chatbot Arena + AgentBench |
| Caching | Redis |
| Blockchain | Ethereum Sepolia (testnet) |
| Smart Contracts | Solidity |

---

## Project Structure

    mnemo/
    ├── agents/         Agent definitions and roles
    ├── data/           Dataset loaders and processing
    ├── graph/          GNN reputation engine
    ├── memory/         Weaviate vector memory layer
    ├── contracts/      Solidity smart contracts
    ├── scripts/        Utility scripts
    ├── notebooks/      Google Colab training notebooks
    ├── main.py         Entry point
    └── README.md
---

## Setup

```bash
git clone https://github.com/Deeksg/Mnemo.git
cd mnemo
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Status

🚧 Active development — Phase 1 (Project Setup)

---

## License

MIT