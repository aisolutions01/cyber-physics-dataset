# cyber-physics-dataset
# On-the-Fly Incremental Learning Framework with Physics-Inspired Constraints

This repository implements a real-time adaptive learning framework that unifies on-the-fly data generation, incremental training, and physics-inspired constraints for continuous self-learning.
It demonstrates how autonomous systems can evolve dynamically from streaming events — without static datasets or retraining cycles.

---

### Project Overview

Traditional machine learning pipelines rely on static, pre-collected datasets, which quickly become outdated in dynamic environments such as cybersecurity and IT operations.
This project introduces a dynamic self-adaptive framework capable of:

1. Generating synthetic incidents on-the-fly
2. Deduplicating recurring patterns automatically
3. Training incrementally on unique events only
4. Performing real-time inference and logging results for audit and comparison

Suitable for **AIOps**, **cybersecurity monitoring**, and **IoT analytics** where continuous learning and fast adaptation are essential.

---

### System Architecture

The framework follows a four-stage pipeline:

<img width="1200" height="627" alt="on-the-fly" src="https://github.com/user-attachments/assets/113c9506-a6d2-4919-8043-356b4f13e357" />

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

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?G(\theta,%20\xi;%20s)%20\rightarrow%20x" />
</p>
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

## Example — Conservation-like Constraint:

For correlated features such as CPU load and network throughput:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?C(x,s)%20=%20\frac{dt}{d(\text{net\_bytes})}%20+%20\alpha%20\text{cpu\_load}%20-%20\beta%20=%200" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\ell_c(C(x,s))%20=%20\left\lVert%20C(x,s)%20\right\rVert_2^2" />
</p>
This enforces smoothness between system metrics, mimicking conservation laws in physical systems — stabilizing the model during real-time adaptation.

---

# Experiments and Results

## Scalability and Stream Volume Analysis

Tested on streams_1k.json (1,000 incidents):

A. Sub-second training and inference per batch (100 events)

B. Accuracy oscillated between 0.48–0.57 across 10 batches

C. Maintained linear scalability with input stream length

The system demonstrated stable adaptation and constant latency under streaming conditions.

| Component           | Description                                     |
| ------------------- | ----------------------------------------------- |
| `data_generator.py` | Synthetic streaming incident generator          |
| `model_example.py`  | Incremental learning and constraint integration |
| `evaluation.py`     | Model performance metrics                       |
| `demo.ipynb`        | Unified demonstration notebook                  |


Example output:

| Metric | Score |
|--------|--------|
| Precision | 0.80 |
| Recall | 0.72 |
| F1-score | 0.68 |
| Accuracy | 0.70 |

Training time per batch: **< 0.40 second**

## Comparative Analysis

| Framework              | Learning Type              | Constraint Support           | Retraining Need |
| ---------------------- | -------------------------- | ---------------------------- | --------------- |
| **River ML**           | Incremental                | ❌ No                         | Partial         |
| **Flink ML**           | Batch + Online             | ❌ No                         | Frequent        |
| **DeepStream SDK**     | Stream Inference           | ❌ No                         | Offline         |
| **Proposed Framework** | **On-the-Fly Incremental** | ✅ **Yes (Physics-Inspired)** | **None**        |

The proposed system unifies data synthesis, constraint enforcement, and model updates in one continuous online loop.

---

# Discussion

A. **Catastrophic Forgetting:** Can be mitigated via replay or consolidation.

B. **Simulator Fidelity:** Realism of generated data influences stability; hybrid real-synthetic input can enhance generalization.

C. **Privacy:** Extendable to encrypted or federated setups for sensitive domains.

---

# Reproducibility

Clone and run:

git clone https://github.com/aisolutions01/cyber-physics-dataset.git

cd cyber-physics-dataset

pip install -r requirements.txt

python data_generator.py

python model_example.py


python evaluation.py

## Environment:

- Python 3.10 / 3.11
- Intel i7 CPU
- 16 GB RAM
- Deterministic random seeds

---

# Citation

Kazem, M. (2025). On-the-Fly Incremental Learning Framework with Physics-Inspired Constraints.

Preprint available at google drive: https://drive.google.com/file/d/1cK7g9NddOiuV7dZxTaFx-t0a4B5Hw4Rj/view?usp=drive_link.

_This preprint will be uploaded to arXiv upon endorsement approval._

---

# Author

Munther Kazem

Computer Scientist • AI Researcher • System Architect

📧 muntherkz2018@gmail.com
