# Embedding Collision Attacks: A Representation-Level Threat to NLP Systems

**Accepted at ESORICS 2026 (European Symposium on Research in Computer Security)**

This repository implements an **Embedding Collision Attack (ECA)** — a representation-level adversarial technique that generates semantically unrelated (off-topic) text which appears highly similar in embedding space to a given target query.

The attack demonstrates how embedding-based systems (e.g., semantic search, RAG, classification) can be manipulated without altering model internals.

---

## Key Idea

Given:
- A **target query** (e.g., cybersecurity content)
- An **off-topic seed** (e.g., unrelated domain text)

ECA modifies the seed using a **Masked Language Model (MLM)** so that:

- Semantic meaning remains largely unrelated
- Embedding similarity with the target becomes **very high**

---

## Method Overview

The attack pipeline consists of:

1. **Embedding Model**
   - Uses Sentence Transformers (default: `all-MiniLM-L6-v2`)
   - Computes similarity between texts

2. **MLM-based Local Search**
   - Uses models like `bert-base-uncased`
   - Iteratively replaces tokens using masked predictions
   - Optimizes cosine similarity with the target

3. **Beam Search Optimization**
   - Keeps top candidate texts per iteration
   - Tracks best-performing adversarial example

4. **Logging (Optional)**
   - Stores step-wise attack progress in `.jsonl` format

---

## Repository Structure

├── ECA.py          
├── logs.jsonl      
└── README.md       

---

## Installation

### 1. Clone the repository
```
git clone https://github.com/bikasaha/ECA.git
cd ECA
```

### 2. Install dependencies
```
pip install torch transformers sentence-transformers nltk numpy
```

### 3. Download NLTK resources
```
import nltk
nltk.download('wordnet')
```

---

## Usage

Run the script directly:
```
python ECA.py
```

### Example
```
target = "The research focuses on developing robust techniques to detect advanced cyber attacks in large-scale computer networks."
off_seed = "The orchestra performed a beautiful symphony in the old concert hall last night."

result = eca_attack(
    target_query=target,
    off_topic_seed=off_seed,
    log_path="logs.jsonl"
)
```

---

## Output
```
{
    "target": "The research focuses on developing robust techniques to detect advanced cyber attacks in large-scale computer networks.",
    "off_topic_seed": "The orchestra performed a beautiful symphony in the old concert hall last night.",
    "seed_sim": -0.05436629801988602,
    "mlm_best_text": "philadelphia security discovered that computer network experienced an big scale attacks monday night.",
    "mlm_best_sim": 0.654070258140564,
}
```

---

## Key Features

- Black-box compatible
- Model-agnostic
- MLM-based optimization
- Beam search
- Logging support

---

## Security Implications

This attack highlights risks in semantic systems like RAG and search pipelines.

---

## License

For research and educational purposes only.
