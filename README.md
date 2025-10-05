# cyber-physics-dataset
# On-the-Fly Incremental Learning Framework with Physics-Inspired Constraints

This repository implements a real-time learning framework that combines **on-the-fly data generation**, **incremental training**, and **physics-inspired constraints** for continuous adaptation.  
It provides a practical demonstration of how autonomous systems can learn from streaming events dynamically — without relying on static datasets.

---

## Project Overview

Traditional machine learning pipelines depend on static, pre-collected datasets.  
This project proposes a **dynamic, self-adaptive framework** capable of:

- Generating synthetic incidents in real time.  
- Deduplicating recurring patterns automatically.  
- Training incrementally on unique events.  
- Performing instant inference and logging outcomes for audit and comparison.  

The system is particularly suited for **AIOps**, **cybersecurity monitoring**, and **IoT analytics**, where continuous learning is critical.

---

## System Architecture

The framework follows a four-stage pipeline:

Stream Ingestion → Deduplication Center → Incremental Trainer → Real-Time Inference → Logging & Comparison


A visual overview is shown below:

<img width="1200" height="627" alt="on-the-fly" src="https://github.com/user-attachments/assets/113c9506-a6d2-4919-8043-356b4f13e357" />
> *Figure 1: Stream ingestion → incremental training → real-time inference → logging and comparison.*

---

## Methodology

### 1. On-the-Fly Data Generator
A lightweight Python-based generator simulates live system incidents in JSON format.  
It produces diverse event types (`Login Failure`, `Malware Detection`, `SLA Violation`, etc.), each labeled with varying severity and source.

Mathematically, the generator is expressed as:
\[
G(\theta, \xi; s) \rightarrow x
\]
where \(s\) is the context, \(\xi\) is stochastic noise, and \(\theta\) represents generator parameters.

---

### 2. Incremental Training & Inference
- Repeated incidents are grouped as one (deduplication).  
- Novel events are used for real-time model updates.  
- Inference occurs immediately, and results are logged.  
- The entire process occurs **on-the-fly** without pre-collected data.

---

### 3. Physics-Inspired Constraints
The learning process is regularized through constraints derived from system dynamics:
\[
\mathcal{L} = \mathbb{E}[L_{\text{task}}] + \lambda \mathbb{E}\left[\sum_j \ell_c(C_j(x,s))\right]
\]
This maintains model stability while adapting continuously.

---

## Experiments and Results

- **Implementation:** Python (data generator + incremental model)  
- **Demo Notebook:** `demo.ipynb`  
- **Dataset:** Synthetic streaming incidents (generated on-the-fly)  
- **Metrics:** Precision, Recall, F1-Score, Accuracy  

Example output:

| Metric | Score |
|--------|--------|
| Precision | 0.80 |
| Recall | 0.72 |
| F1-score | 0.68 |
| Accuracy | 0.70 |

Training time per batch: **< 1 second**

---

## 💡 Discussion

- **Catastrophic Forgetting:** Incremental learning may forget old patterns over time — mitigated using replay or consolidation strategies.  
- **Simulator Fidelity:** The realism of synthetic data affects generalization.  
- **Privacy:** Extendable to federated or privacy-preserving systems.

---

## Reproducibility

All experiments can be reproduced easily.

### Steps:
```bash
git clone https://github.com/aisolutions01/cyber-physics-dataset.git
cd stream-incremental-ai
pip install -r requirements.txt
python data_generator.py
python model_example.py
python evaluation.py

The environment was tested on:

Python 3.10
Intel i7 CPU
16 GB RAM
Random seeds are fixed for deterministic behavior.

---

Future Work

Integrate Reinforcement Learning for adaptive model updates.
Deploy within large-scale systems (e.g., Spark Streaming, Flink).
Use digital twin simulations for richer synthetic data.
Extend to real-world event streams and industrial IoT.

Kazem, M. (2025). On-the-Fly Incremental Learning Framework with Physics-Inspired Constraints.
Preprint available at arXiv: [link pending submission].

Author

Munther Kazem
Computer Scientist | AI Researcher | System Architect
