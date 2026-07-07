# AdaRAG: Query-Adaptive Retrieval Budgeting for Retrieval-Augmented Generation

AdaRAG is an adaptive retrieval budgeting framework that estimates query complexity before retrieval and dynamically allocates the retrieval budget. Instead of retrieving a constant Top-k for every query, AdaRAG computes a Retrieval Demand Score (RDS) using linguistic, semantic, and structural query features, selecting the optimal retrieval budget (Top-3, Top-7, or Top-12).

## Repository Structure

```
AdaRAG/
│
├── config.py
├── run.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── corpus/
│   ├── evaluation/
│   └── results/
│
├── models/
│
└── src/
    ├── utils.py
    ├── constants.py
    ├── embedder.py
    ├── retriever.py
    ├── indexer.py
    ├── llm.py
    ├── analyzer.py
    ├── budget.py
    ├── pipeline.py
    └── evaluator.py
```

## Features

- **Query Complexity Analyzer (`src/analyzer.py`)**: Extracts lexical, structural, and semantic features using spaCy.
- **Retrieval Demand Estimator (`src/budget.py`)**: Computes the Retrieval Demand Score (RDS) to determine the adaptive retrieval budget.
- **FAISS Retriever (`src/retriever.py`)**: Stores document chunk embeddings and retrieves context using cosine similarity.
- **OpenAI LLM Wrapper (`src/llm.py`)**: Utilizes OpenAI GPT models for answer generation, with built-in token tracking and a local mock fallback.
- **Evaluation Suite (`src/evaluator.py`)**: Compares AdaRAG against fixed Top-k baselines, computes Exact Match, F1 score, latency, and cost, and performs feature ablation studies.

## Setup Instructions

1. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the required spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. Create a `.env` file in the root directory and configure your OpenAI API Key:
   ```env
   OPENAI_API_KEY=your-api-key-here
   ```

## Execution Commands

### 1. Build Vector Index
Process the documents in the corpus and build the FAISS index:
```bash
python run.py --mode index
```

### 2. Run Evaluation Experiments
Compare AdaRAG against Fixed Top-5 and Fixed Top-10 baselines:
```bash
python run.py --mode evaluate
```
This writes the comparison tables to `data/results/experiment_results.csv` and saves the visual comparison plot to `data/results/evaluation_comparison.png`.

### 3. Run Ablation Study
Measure the performance impact of removing individual feature categories:
```bash
python run.py --mode ablate
```
Writes the ablation table to `data/results/ablation_results.csv`.

### 4. Interactive Mode
Run queries manually to see the computed complexity scores and selected budget dynamically:
```bash
python run.py --mode interactive
```