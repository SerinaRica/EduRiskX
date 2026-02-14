# EduRiskX: Hybrid Deep Learning and F-Logic Reasoning for Early At-Risk Prediction

EduRiskX is a comprehensive framework for early detection of at-risk students in online learning environments (OULAD). It combines advanced deep learning models (Transformer/PatchTST/iTransformer) with interpretable symbolic reasoning (F-Logic) to provide both high predictive accuracy and actionable explanations.

## 📂 Project Structure

The project is organized into the following modules:

```
eduriskX/
├── predictors/          # Deep Learning Model Architectures
│   ├── hybrid.py        # EduRiskX Hybrid Predictor (Neural + F-Logic)
│   ├── transformer.py   # Optimized Transformer with Temporal Attention
│   ├── patchtst.py      # PatchTST Implementation
│   ├── itransformer.py  # iTransformer Implementation
│   └── ...
├── reasoning/           # Symbolic Reasoning Engine
│   ├── reasoner.py      # F-Logic Reasoner
│   └── ...
├── src/                 # Core Utilities & Rule Mining
│   ├── rule_extraction.py # Theory-Aligned Rule Mining
│   ├── 04_theory_alignment.py # Theoretical Alignment Scoring
│   ├── 01_data_preprocesing.py # Data Preprocessing
│   ├── 02_feature_engineering.py # Feature Engineering
│   └── ...
├── experiment/          # Experiment & Benchmark Scripts
│   ├── run_weekly_benchmark.py # Weekly Early Detection Benchmark
│   ├── run_significance_test.py # Statistical Significance Testing (Paired t-test)
│   └── ...
├── data/                # Data Files
│   ├── raw/             # Raw OULAD Data (Download Required)
│   ├── processed/       # Preprocessed Feature Data (Parquet)
│   └── config/          # Configuration Files (Theory Keywords)
├── outputs/             # Generated Rules, Plots, and Reports
│   └── rules/           # English Rule Definitions (JSON)
├── requirements.txt     # Python Dependencies
├── generate_paper_plots.py # Plot Generation for Research Paper
└── main.py              # Main Entry Point
```

## 🚀 Installation

1.  **Prerequisites**: Python 3.9 or higher.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🔄 Data Preparation & Full Workflow

### 1. Download OULAD Dataset
The **Open University Learning Analytics Dataset (OULAD)** is required to run the full pipeline.
1.  Visit the [OULAD repository](https://analyse.kmi.open.ac.uk/open_dataset) or [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/349/student+performance).
2.  Download the CSV files.
3.  Extract them into `data/raw/` directory.
    *   `data/raw/studentInfo.csv`
    *   `data/raw/studentVle.csv`
    *   `data/raw/studentAssessment.csv`
    *   `data/raw/assessments.csv`
    *   `data/raw/studentRegistration.csv`
    *   `data/raw/vle.csv`
    *   `data/raw/courses.csv`

### 2. Preprocessing & Feature Engineering
Convert raw CSVs into weekly time-series features.
```bash
# 1. Clean raw data and handle missing values
python src/01_data_preprocesing.py

# 2. Generate weekly features (aggregated by student-week)
python src/02_feature_engineering.py
```
*Output*: `data/processed/weekly_features.parquet` (This file is already included for quick start, but you can regenerate it from raw data).

## 🛠 Usage

### 1. Rule Mining (Theory-Aligned)
Extracts interpretable risk rules from data and aligns them with educational theories (Engagement, Self-Efficacy, etc.).
```bash
python src/rule_extraction.py
```
*Output*: `outputs/rules/enhanced_rules.json` (English version)

### 2. Weekly Benchmark (Early Detection)
Runs the weekly prediction benchmark (Weeks 5-38) comparing EduRiskX against baselines (LSTM, CNN, PatchTST, iTransformer).
```bash
python experiment/run_weekly_benchmark.py
```
*Output*: Metrics and plots in `outputs/experiments/weekly_benchmark/`

### 3. Statistical Significance Test
Conducts paired t-tests across 5 random seeds to validate the superiority of EduRiskX.
```bash
python experiment/run_significance_test.py
```
*Output*: Significance tables and p-values.

### 4. Generate Paper Plots
Generates publication-quality figures (Radar Charts, PR Curves, Early Detection Timeline).
```bash
python generate_paper_plots.py
```

## ✨ Key Features

*   **Hybrid Architecture**: Combines neural probabilities with symbolic rule scores using a "Probabilistic OR" logic (`Final_Risk = Neural + Rule - Neural*Rule`).
*   **Theory-Aligned Explanations**: Rules are not just data patterns but are mapped to educational theories (e.g., "Low Engagement" -> Engagement Theory).
*   **Early Detection**: Optimized for high accuracy in early weeks (Weeks 5-15) of the semester.
*   **Ablation Studies**: Rigorous analysis of each component's contribution (Temporal Attention, Class-Weighted Loss, F-Logic).
*   **SOTA Baselines**: Benchmarked against state-of-the-art time-series models (PatchTST, iTransformer).

## 📊 Data
The system uses preprocessed weekly feature data stored in `data/processed/weekly_features.parquet`. This file contains time-series features for students across different courses.
