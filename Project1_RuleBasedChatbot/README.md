# Rule-Based AI Chatbot - DecodeLabs Foundation Phase

## Project Overview
This repository contains the Module 01 deliverable for the DecodeLabs Artificial Intelligence Industrial Training Kit. It is a deterministic, rule-based AI chatbot engineered in Python. Rather than utilizing probabilistic deep learning models, this system implements a strict "White Box" IPO (Input, Process, Output) architecture. It demonstrates proficiency in control flow, memory-optimized data handling using `pandas`, and modular system design.

## Features
*   **Continuous Event Loop:** Maintains a persistent state until an explicit kill command is issued.
*   **O(1) Intent Matching:** Utilizes hash maps for instant retrieval of static greetings and commands.
*   **Dynamic Data Querying:** Parses natural language variations to execute analytical operations against a 1,200+ record dataset.
*   **Memory Optimization:** Strictly adheres to a single-load principle, avoiding recursive DataFrame duplication and utilizing in-place boolean masking.
*   **Deterministic Safety:** Zero risk of "hallucinations." All responses are explicitly defined or mathematically derived from the source data.

## Folder Structure
```text
rule_based_chatbot/
│
├── data/
│   └── Dataset for Data Analytics.xlsx
│
├── src/
│   ├── main.py            # Execution layer and event loop
│   ├── chatbot.py         # Control flow and routing logic
│   ├── query_engine.py    # Analytical data operations
│   ├── data_loader.py     # Memory-safe file I/O
│   └── responses.py       # Static strings and O(1) hash maps
│
├── requirements.txt
└── README.md