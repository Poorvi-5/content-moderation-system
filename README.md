# 🚀 Real-Time Multimodal Content Moderation System with MLOps and CI/CD Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green.svg)
![Docker](https://img.shields.io/badge/Docker-24.0-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.3-purple.svg)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-black.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Production-grade multimodal content moderation system with DistilBERT-based text detection, MobileNetV2/ResNet computer vision, MLOps pipelines, CI/CD automation, model monitoring, drift detection, and continuous retraining loop.**

---

## 📖 Table of Contents
- [Project Description](#-project-description)
- [System Analogy 🎭](#-system-analogy-)
- [System Architecture](#-system-architecture)
- [CI/CD Pipeline](#-cicd-pipeline)
- [MLOps Continuous Loop](#-mlops-continuous-loop)
- [Models Used](#-models-used)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Setup & Installation](#-setup--installation)
- [Running the System](#-running-the-system)
- [API Endpoints & Usage](#-api-endpoints--usage)
- [MLflow Tracking](#-mlflow-tracking)
- [Monitoring & Drift Detection](#-monitoring--drift-detection)
- [CI/CD in Action](#-cicd-in-action)
- [Future Improvements](#-future-improvements)
- [Resume One-Liner](#-resume-one-liner)

---

## 🎯 Project Description

**What it does:** This system moderates user-generated content in **real time** across three modalities — **text, images, and video**. It uses fine-tuned transformer and CNN models to score content for toxicity, violence, or policy violations, makes an **allow/review/block** decision, logs every prediction, monitors for data drift, and automatically triggers **continuous retraining** when model performance degrades.

**Why it matters:** Online platforms receive billions of content pieces daily. Manual moderation doesn't scale, and static models become outdated as language and user behavior evolve. This system closes the loop between **prediction, monitoring, feedback, and retraining** — enabling self-healing AI moderation.

**Key capabilities:**
- ⚡ Sub-100ms inference per text/image on CPU
- 🎥 Video frame sampling with temporal smoothing
- 📊 Real-time monitoring metrics (block rate, latency, drift)
- 🔁 Automatic retraining triggered by drift detection
- 🧪 Human-in-the-loop feedback for continuous improvement

---

## 🎭 System Analogy (Nightclub Bouncer)

Imagine a high-end nightclub with **one head bouncer** and **three specialized assistants**:

| Assistant | Real-World Role | AI Equivalent |
|-----------|----------------|----------------|
| **ID Checker** | Checks if someone is banned or causing past trouble | Text moderation (past toxic comments, hate speech) |
| **Appearance Checker** | Looks for prohibited attire, offensive symbols | Image classification (violence, nudity, gore) |
| **Behavior Watcher** | Observes how someone acts over 10-30 seconds | Video analysis (aggressive gestures, repeated violations) |

**The Bouncer (Risk Scoring Engine)** combines all three scores, applies rules (e.g., "If any score > 0.9 → BLOCK"), and decides: **Allow entry**, **Review with manager**, or **Block immediately**.

**The Security Log (Monitoring)** records every decision, tracks how often blocks happen, and alerts if suddenly 50% more people are blocked (drift).

**The Weekly Training (Retraining Loop)** takes feedback from wrong decisions and updates the assistants' playbooks.

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph INPUT["📥 USER CONTENT INPUT"]
        A1["📝 Text"]
        A2["🖼️ Image"]
        A3["🎥 Video"]
    end

    subgraph API["🌐 API GATEWAY"]
        B["FastAPI + Uvicorn<br/>Async Endpoints"]
    end

    subgraph MODELS["🤖 MODELS LAYER"]
        C1["📝 DistilBERT<br/>Text Classifier<br/>6 toxicity classes"]
        C2["🖼️ ResNet18<br/>Image Classifier<br/>violence/nudity/gore"]
        C3["🎥 MobileNetV2<br/>Video Frame Sampler<br/>+ Temporal Smoothing"]
    end

    subgraph ENGINE["⚙️ DECISION ENGINE"]
        D1["Risk Scoring Engine<br/>Weighted ensemble<br/>Configurable thresholds"]
        D2["Decision Engine<br/>allow / review / block"]
    end

    subgraph OUTPUT["✅ OUTPUT"]
        E1["🟢 ALLOW<br/>Content published"]
        E2["🟡 REVIEW<br/>Human moderator"]
        E3["🔴 BLOCK<br/>Content rejected"]
    end

    subgraph MONITORING["📊 MONITORING & OPS"]
        F1[("SQLite DB<br/>Prediction logs")]
        F2["Drift Detection<br/>Score / Latency / Volume"]
        F3["Retraining Trigger<br/>>20% drift → retrain"]
    end

    A1 --> B
    A2 --> B
    A3 --> B
    
    B --> C1
    B --> C2
    B --> C3
    
    C1 --> D1
    C2 --> D1
    C3 --> D1
    
    D1 --> D2
    D2 --> E1
    D2 --> E2
    D2 --> E3
    
    E2 -.->|"Human Feedback"| F1
    D1 --> F1
    D2 --> F1
    
    F1 --> F2
    F2 -->|"Drift > Threshold"| F3
    F3 -->|"Trigger Retraining"| C1
    F3 -->|"Trigger Retraining"| C2
    F3 -->|"Trigger Retraining"| C3

    style INPUT fill:#e1f5fe
    style API fill:#fff3e0
    style MODELS fill:#f3e5f5
    style ENGINE fill:#e8f5e9
    style OUTPUT fill:#e0f2f1
    style MONITORING fill:#ffebee
