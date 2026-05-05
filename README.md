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
---




##🔁 CI/CD Pipeline
flowchart LR
    subgraph DEV["👨‍💻 DEVELOPER"]
        PUSH["git push origin main<br/>or Pull Request"]
    end

    subgraph GHA["⚙️ GITHUB ACTIONS"]
        direction TB
        TRIGGER["🚀 Workflow Triggered"]
        
        subgraph TEST["🧪 TEST STAGE"]
            T1["Unit Tests<br/>pytest tests/"]
            T2["Linting<br/>flake8, black"]
            T3["Model Validation<br/>Load + dummy infer"]
        end
        
        subgraph BUILD["📦 BUILD STAGE"]
            B1["Build Docker Image<br/>docker build"]
            B2["Tag Image<br/>git-sha + latest"]
            B3["Security Scan<br/>trivy"]
        end
        
        subgraph DEPLOY["🚀 DEPLOY STAGE"]
            D1["Push to Registry<br/>GHCR / Docker Hub"]
            D2["Deploy to Staging<br/>Smoke tests"]
            D3["Deploy to Production<br/>(main branch only)"]
        end
        
        TRIGGER --> TEST
        TEST -->|"All tests pass"| BUILD
        BUILD -->|"Image ready"| DEPLOY
    end

    subgraph NOTIFY["📢 NOTIFICATION"]
        N1["Slack Alert<br/>Deployment status"]
        N2["MLflow Registry<br/>Update model version"]
    end

    PUSH --> GHA
    DEPLOY --> NOTIFY

    style DEV fill:#e3f2fd
    style GHA fill:#f3e5f5
    style TEST fill:#e8f5e9
    style BUILD fill:#fff3e0
    style DEPLOY fill:#e0f2f1
    style NOTIFY fill:#fce4ec


##CI/CD Pipeline Steps Detail:
Stage	Actions	Triggers	Success Criteria
Test	Unit tests, linting, model validation	Every push & PR	All tests pass, no lint errors
Build	Docker build, tagging, security scan	Push to main or PR	Image builds, no critical CVEs
Deploy	Push to registry, staging deploy, prod deploy	Main branch only	Smoke tests pass, health check OK
Notify	Slack alert, MLflow registry update	After deploy	Webhook succeeds


##🔄 MLOps Continuous Loop
flowchart TB
    subgraph DATA["📊 DATA PIPELINE"]
        D1["Raw Data<br/>Civil Comments<br/>CIFAR-10"]
        D2["DVC Versioning<br/>Data preprocessing<br/>Train/val splits"]
        D3[("Feature Store<br/>Processed tensors")]
    end

    subgraph TRAIN["🧠 MODEL TRAINING"]
        T1["DistilBERT Fine-tuning<br/>or ResNet18 Training"]
        T2["MLflow Tracking<br/>Params, metrics, artifacts"]
        T3["Model Registry<br/>Staging / Production"]
    end

    subgraph SERVE["🚀 MODEL SERVING"]
        S1["FastAPI Server<br/>Load models at startup"]
        S2["Inference Endpoints<br/>/moderate/text|image|video"]
    end

    subgraph MONITOR["📈 MONITORING"]
        M1[("Prediction Logs<br/>SQLite Database")]
        M2["Metrics Computation<br/>Block rate, latency, volume"]
        M3["Drift Detection<br/>Score / Latency / Decision"]
    end

    subgraph FEEDBACK["🔄 FEEDBACK LOOP"]
        F1["Human Feedback API<br/>/monitoring/feedback"]
        F2["Feedback Labels<br/>Corrected decisions"]
        F3["Drift >20%?"]
    end

    subgraph RETRAIN["🔁 RETRAINING"]
        R1["Fetch New Data<br/>Last 7 days + feedback"]
        R2["Create Dataset<br/>Augmented + relabeled"]
        R3["Launch Training Job<br/>Same hyperparameters"]
        R4["Validate Model<br/>A/B test in shadow mode"]
    end

    D1 --> D2
    D2 --> D3
    
    D3 --> T1
    T1 --> T2
    T2 --> T3
    
    T3 --> S1
    S1 --> S2
    
    S2 --> M1
    M1 --> M2
    M2 --> M3
    
    F1 --> F2
    F2 --> M1
    
    M3 --> F3
    F3 -->|No| MONITOR
    F3 -->|Yes| R1
    
    R1 --> R2
    R2 --> R3
    R3 --> R4
    R4 -->|"Better than current?"| T3
    R4 -->|"No improvement"| MONITOR

    style DATA fill:#e1f5fe
    style TRAIN fill:#f3e5f5
    style SERVE fill:#e8f5e9
    style MONITOR fill:#fff3e0
    style FEEDBACK fill:#ffebee
    style RETRAIN fill:#fce4ec

##🤖 Inference Request-Response Flow (Detailed)
sequenceDiagram
    autonumber
    participant Client
    participant API as FastAPI Load Balancer
    participant Auth as API Key Auth
    participant TextModel as DistilBERT Model
    participant ImageModel as ResNet18 Model
    participant VideoModel as MobileNetV2 Model
    participant Risk as Risk Scoring Engine
    participant Decision as Decision Engine
    participant DB as SQLite DB
    participant Monitor as Drift Detector

    Client->>API: POST /moderate/all (text + image)
    API->>Auth: Validate API key
    Auth-->>API: Authorized
    
    par Parallel Inference
        API->>TextModel: Predict text toxicity
        TextModel-->>API: {toxic:0.94, insult:0.91}
    and
        API->>ImageModel: Predict image safety
        ImageModel-->>API: {class:"violence", conf:0.96}
    and
        API->>VideoModel: (if video provided)
        VideoModel-->>API: {avg_conf:0.89}
    end
    
    API->>Risk: Aggregate scores (weights: text=0.3, image=0.5, video=0.2)
    Risk-->>API: Risk Score = 0.87
    
    API->>Decision: Apply rules: Risk>0.8 → BLOCK, 0.5-0.8 → REVIEW
    Decision-->>API: Decision = "BLOCK"
    
    API->>DB: Log prediction {timestamp, scores, decision, latency}
    DB-->>API: Logged
    
    API-->>Client: 200 OK {is_violation:true, decision:"block"}
    
    Note over DB,Monitor: Drift detection runs every 6 hours
    Monitor->>DB: Query last 6h vs previous 6h
    alt Drift > 20%
        Monitor->>API: Trigger retraining alert
    end


##📊 Data Pipeline (ETL) Detailed Flow
flowchart LR
    subgraph SOURCES["📁 DATA SOURCES"]
        S1["Google Civil Comments<br/>1.8M rows • CSV"]
        S2["CIFAR-10<br/>60K images • PNG"]
        S3["Custom Dataset<br/>(future)"]
    end

    subgraph EXTRACT["⬇️ EXTRACT"]
        E1["download_data.py<br/>wget / Kaggle API"]
        E2["Validate checksums<br/>Verify integrity"]
    end

    subgraph TRANSFORM["🔄 TRANSFORM"]
        direction TB
        T1_TEXT["Text Pipeline:<br/>• Lowercase<br/>• Remove special chars<br/>• DistilBERT tokenization<br/>• Pad/truncate to 512"]
        T2_IMAGE["Image Pipeline:<br/>• Resize 224×224<br/>• Normalize (ImageNet stats)<br/>• ToTensor<br/>• Save as .pt"]
        T3_VIDEO["Video Pipeline:<br/>• Extract frames (5 fps)<br/>• Resize each frame<br/>• Stack tensors"]
    end

    subgraph LOAD["💾 LOAD"]
        L1[("Processed Text<br/>.parquet + .txt")]
        L2[("Processed Images<br/>.pt tensors")]
        L3[("Video Frames<br/>.npy arrays")]
    end

    subgraph VERSION["📌 VERSION CONTROL"]
        V1["DVC Track<br/>dvc add data/processed"]
        V2["Remote Storage<br/>S3 / GCS / Drive"]
    end

    S1 --> E1
    S2 --> E1
    S3 --> E1
    
    E1 --> E2
    E2 --> T1_TEXT
    E2 --> T2_IMAGE
    E2 --> T3_VIDEO
    
    T1_TEXT --> L1
    T2_IMAGE --> L2
    T3_VIDEO --> L3
    
    L1 --> V1
    L2 --> V1
    L3 --> V1
    
    V1 --> V2

    style SOURCES fill:#e1f5fe
    style EXTRACT fill:#fff3e0
    style TRANSFORM fill:#f3e5f5
    style LOAD fill:#e8f5e9
    style VERSION fill:#e0f2f1

##🧠 Model Training Pipeline (Detailed)

flowchart TB
    subgraph INPUT["📊 TRAINING DATA"]
        I1["Train Set<br/>80% of data"]
        I2["Validation Set<br/>10% of data"]
        I3["Test Set<br/>10% of data"]
    end

    subgraph CONFIG["⚙️ CONFIGURATION"]
        C1["Model:<br/>DistilBERT / ResNet18"]
        C2["Hyperparameters:<br/>lr, batch_size, epochs"]
        C3["Loss Function:<br/>CrossEntropy<br/>+ class weights"]
    end

    subgraph TRAINING["🔄 TRAINING LOOP"]
        direction TB
        T1["Initialize Model<br/>Load pretrained weights"]
        T2["Forward Pass<br/>Compute predictions"]
        T3["Calculate Loss<br/>Backpropagate"]
        T4["Optimizer Step<br/>Update weights"]
        T5["Validation<br/>Every N epochs"]
    end

    subgraph TRACKING["📈 MLFLOW TRACKING"]
        M1["Log Parameters<br/>lr, batch_size, model"]
        M2["Log Metrics<br/>loss, accuracy, f1"]
        M3["Log Artifacts<br/>Model weights, plots"]
    end

    subgraph OUTPUT["💾 OUTPUT"]
        O1["Best Model<br/>.bin / .pth file"]
        O2["Model Registry<br/>Staging tag"]
        O3["Confusion Matrix<br/>PR Curves"]
    end

    I1 --> TRAINING
    I2 --> TRAINING
    I3 -->|"Final evaluation"| O3
    
    C1 --> T1
    C2 --> T2
    C3 --> T3
    
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> T5
    T5 -->|"If val loss improves"| T1
    T5 -->|"Save checkpoint"| O1
    
    T1 -.-> M1
    T2 -.-> M2
    T3 -.-> M2
    O1 -.-> M3
    O1 --> O2

    style INPUT fill:#e1f5fe
    style CONFIG fill:#fff3e0
    style TRAINING fill:#f3e5f5
    style TRACKING fill:#e8f5e9
    style OUTPUT fill:#e0f2f1
