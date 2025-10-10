## cyber-physics-dataset
# On-the-Fly Incremental Learning Framework with Physics-Inspired Constraints

This repository implements a real-time adaptive learning framework that unifies on-the-fly data generation, incremental training, and physics-inspired constraints for continuous self-learning.
It demonstrates how autonomous systems can evolve dynamically from streaming events — without static datasets or retraining cycles.
---

## Project Overview

Traditional machine learning pipelines rely on static, pre-collected datasets, which quickly become outdated in dynamic environments such as cybersecurity and IT operations.
This project introduces a dynamic self-adaptive framework capable of:

1. Generating synthetic incidents on-the-fly
2. Deduplicating recurring patterns automatically
3. Training incrementally on unique events only
4. Performing real-time inference and logging results for audit and comparison

Suitable for **AIOps**, **cybersecurity monitoring**, and **IoT analytics** where continuous learning and fast adaptation are essential.

---

## System Architecture

The framework follows a four-stage pipeline:

<img width="1200" height="627" alt="on-the-fly" src="https://github.com/user-attachments/assets/113c9506-a6d2-4919-8043-356b4f13e357" />
*Figure 1: Stream ingestion → incremental training → real-time inference → logging and comparison.*

---

## Methodology

### 1. On-the-Fly Data Generator
A Python-based generator continuously simulates operational incidents in JSON format:
{
  "timestamp": "2025-10-07T12:00:00",
  "incident_type": "login_fail",
  "severity": 2,
  "source": "endpoint-01",
  "cpu_load": 0.233,
  "net_bytes": 19766
}

G(θ,ξ;s)→x
where 𝑠 is the context, 𝜉 stochastic noise, and 𝜃 generator parameters.

---

### 2. Incremental Training & Inference
A. Deduplicate repeating incidents
B. Train model incrementally on unseen samples
C. Predict incident priority or class in real time
D. Log every event and decision for audit comparison

All processes occur on-the-fly — no batch storage or pre-collected data.

---

### 3. Physics-Inspired Constraints
The training loss integrates operational or physical constraints:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\lambda%20\mathbb{E}\left[\sum_j\ell_c(C_j(x,s))\right]%20+%20\mathbb{E}[L_{\text{task}}]%20=%20\mathcal{L}" />
</p>

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
