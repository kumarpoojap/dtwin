# Autonomous Thermal Orchestration in GPU‑Dense Servers  
## A Node‑Level Agentic Digital Twin for Predictive Throttle Avoidance

### Proposal for M.Tech Dissertation (WILP)
**Student:** Pooja  
**Program:** M.Tech (AI / ML), BITS Pilani – Work Integrated Learning  
**Proposed Supervisor:** Dr. Tabet Said  

---

## 1. Introduction and Motivation

The rapid growth of AI innovation and deployment has led to unprecedented demands on computing infrastructure, particularly in GPU‑dense servers supporting large‑scale LLM training and bursty inference workloads. These systems operate at extreme power densities, significantly increasing energy consumption, operational cost, and thermal stress. In practice, thermal management in such environments is often reactive, relying on threshold‑based fan curves or conservative safety margins.

One of the most visible symptoms of this challenge is frequent thermal throttling triggered by rapid temperature transients. Sub‑optimal cooling configurations and workload orchestration lead to degraded performance, increased energy usage, and accelerated aging of expensive GPU hardware. Improving thermal orchestration therefore has a threefold impact: preserving performance, reducing energy cost, and extending infrastructure longevity.

This dissertation explores the use of **agentic digital twins**—combining predictive physical modeling with autonomous decision‑making—to move thermal control from a reactive to a predictive paradigm at the server (node) level.

---

## 2. Problem Statement

Existing thermal control mechanisms in GPU servers are predominantly reactive and locally optimized. They do not anticipate future thermal states, nor do they explicitly optimize the trade‑off between performance, energy efficiency, and hardware safety under highly dynamic AI workloads.

There is a need for a **lightweight, causal, and explainable digital twin** that can:
- predict near‑term thermal evolution,
- reason about the future impact of control actions,
- and autonomously orchestrate cooling actions while respecting safety constraints.

---

## 3. Proposed Solution

This dissertation proposes a **Node‑Level Agentic Digital Twin (ADT)** for predictive thermal orchestration in GPU‑dense servers.

The ADT integrates:
- a **physics‑guided surrogate thermal model** capable of short‑horizon prediction, and
- a **safe reinforcement learning (RL) controller** that uses these predictions to proactively avoid thermal throttling.

The system operates in a closed cyber‑physical loop:

```
Telemetry → State Construction → Twin Prediction → RL Decision → Safety Filter → Actuation → Physical Response
```

---

## 4. Digital Twin Architecture

### 4.1 Surrogate Thermal Model

The digital twin uses a **lumped‑parameter RC‑based thermal model** as its core surrogate. This model captures:
- heat accumulation from GPU power,
- cooling effects from actuators (fan or pump),
- passive dissipation to the environment,
- and thermal inertia.

To improve fidelity while preserving causality and stability, an **optional PINN‑lite residual** is added, where a shallow neural network learns deviations from the RC baseline while being constrained by physical residuals.

### 4.2 Agentic Control Layer

A **model‑based RL agent** (Soft Actor‑Critic) operates on top of the surrogate. The agent:
- observes the current thermal state,
- reasons over predicted temperature trajectories (30–90 seconds),
- selects continuous cooling actions (fan/pump speed),
- optimizes a multi‑objective reward balancing thermal safety, energy usage, and control smoothness.

### 4.3 Safety and Governance Layer

A lightweight safety layer enforces:
- action bounds,
- rate limits on actuation,
- hard thermal constraints,
- and rule‑based fallback behavior.

This ensures reliable and trustworthy operation suitable for cyber‑physical systems.

---

## 5. Data Sources

### 5.1 Synthetic Thermal Dataset (Primary)

A synthetic dataset is generated using an RC‑based thermal simulator with:
- realistic GPU workload patterns,
- ambient or coolant temperature variation,
- cooling actuation signals,
- and sensor noise.

This dataset provides controlled, reproducible ground truth for:
- surrogate calibration,
- surrogate evaluation,
- RL environment simulation,
- and stress testing.

### 5.2 Optional Real PowerEdge Telemetry (Supplementary)

If available, real telemetry from a PowerEdge GPU server may be used **only** to calibrate and validate the surrogate model for a single GPU thermal path. This does not change the scope or architecture and does not involve live hardware control.

---

## 6. Methodology Overview

1. **Surrogate Construction and Calibration**
   - Fit RC model coefficients using synthetic/proxy data.
   - Train PINN‑lite residual with combined data and physics loss.
   - Evaluate single‑step and multi‑step prediction accuracy.

2. **Multi‑Step Rollout Evaluation**
   - Validate surrogate stability across 10–90 second horizons.
   - Analyze drift, error accumulation, and causality.

3. **RL Environment and Training**
   - Wrap the surrogate as a Gym‑style environment.
   - Train a model‑based RL policy using curriculum learning and domain randomization.

4. **Baseline Controllers**
   - Static fan curves.
   - Threshold‑based control.
   - Lightweight Model Predictive Control (MPC) using the same surrogate.

5. **Closed‑Loop Evaluation**
   - Compare RL against baselines under realistic and stress conditions.

---

## 7. Causality and Explainability

- **Causality:** The RC‑based surrogate explicitly encodes cause‑effect relationships between workload, power, cooling actions, and temperature evolution. This prevents non‑physical behavior and supports counterfactual reasoning.
- **Explainability:** Model parameters and predicted temperature trajectories provide interpretable justification for control actions. The safety layer further enhances transparency and trustworthiness.

---

## 8. Experiments and Evaluation

### 8.1 Surrogate Evaluation
- RC vs PINN‑lite vs pure ML (RF/MLP) baselines.
- One‑step and multi‑step prediction error.
- Stability under long rollouts.

### 8.2 Control Evaluation
- RL vs static fan curve.
- RL vs threshold control.
- RL vs MPC baseline.

### 8.3 Stress Scenarios
- Workload bursts.
- Ambient temperature shifts.
- Sensor noise and actuator imperfections.

### 8.4 Metrics

**Primary:**
- Throttling event reduction.
- Thermal headroom.
- Time‑to‑cool.

**Secondary:**
- Cooling energy usage.
- Actuation stability.
- Predictive behavior (early intervention).

---

## 9. Expected Contributions

- A **reproducible, safety‑aware node‑level Agentic Digital Twin** for GPU thermal orchestration.
- Demonstration of predictive throttle avoidance with equal or lower energy cost than reactive baselines.
- Ablation study comparing RC vs PINN‑lite surrogates.
- Analysis of shielded vs unshielded RL control.
- A scalable architectural template for higher‑level datacenter autonomy.

---

## 10. Scope and Feasibility

The work is intentionally scoped to a **single node / GPU thermal subsystem**, enabling full implementation, experimentation, and evaluation within the M.Tech timeline. While lightweight in computation, the methodology reflects real‑world cyber‑physical control practices and is designed to scale to rack‑ and aisle‑level orchestration.

---

## 11. Timeline (Indicative)

- **Mar–Apr:** Surrogate modeling and rollout evaluation  
- **May:** RL training and baseline comparisons  
- **Jun:** Mid‑semester report and review  
- **Jul:** Stress testing, analysis, and writing  
- **Aug:** Final submission and viva‑voce  

---

## 12. Conclusion

By integrating physics‑guided surrogate modeling with safe, predictive reinforcement learning, this dissertation aims to demonstrate how agentic digital twins can enable autonomous, energy‑efficient, and trustworthy thermal orchestration in modern GPU‑dense infrastructure.
