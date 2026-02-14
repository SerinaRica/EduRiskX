# EduRiskX: Neuro-Symbolic Student Risk Prediction Framework

EduRiskX (Educational Risk eXplainer) is a comprehensive framework designed to predict at-risk students in online learning environments. It combines the predictive power of modern Deep Learning models (such as Transformers) with the interpretability of Symbolic Reasoning (F-Logic) to provide accurate, explainable, and actionable insights.

## 🚀 Key Features

*   **Hybrid Neuro-Symbolic Architecture**: Integrates deep neural networks (Transformer, PatchTST, iTransformer) with an F-Logic reasoning engine. The neural model provides a base risk probability, which is then refined and explained by the symbolic reasoner.
*   **Educational Theory Alignment**: Automatically aligns data-driven rules with established educational theories (e.g., Self-Regulated Learning, Social Cognitive Theory) using semantic embeddings (SentenceTransformers).
*   **Explainable Risk Prediction**: Beyond just a risk score, EduRiskX provides:
    *   **Triggered Rules**: Specific conditions that led to the risk assessment.
    *   **Severity Levels**: Classified risk levels (Low, Medium, High, Critical).
    *   **Theoretical Basis**: Which educational theories explain the student's behavior.
*   **Actionable Interventions**: Generates specific intervention suggestions based on the identified risk factors.

## 📂 Project Structure

```
EduRiskX/
├── data/
│   ├── raw/                  # Place OULAD dataset files here
│   ├── processed/            # Preprocessed features and statistics
│   └── config/               # Configuration files (e.g., theory keywords)
├── predictors/               # Neural network models
│   ├── base_predictor.py     # Base class for predictors
│   ├── hybrid.py             # Hybrid Neuro-Symbolic predictor
│   ├── patchtst.py           # PatchTST implementation
│   ├── itransformer.py       # iTransformer implementation
│   └── transformer.py        # Standard Transformer implementation
├── reasoning/                # Symbolic reasoning engine
│   ├── reasoner.py           # Main reasoning logic
│   ├── evidence_aggregation.py # Belief aggregation logic
│   └── intervention_mapper.py  # Mapping risks to interventions
├── src/                      # Data processing and core logic
│   ├── 01_data_preprocesing.py # Data loading and cleaning
│   ├── 04_theory_alignment.py  # Semantic alignment with theories
│   ├── flogic_parser.py      # F-Logic rule parsing
│   └── rule_extraction.py    # Extracting rules from data
├── experiment/               # Experiment scripts
│   ├── run_comprehensive_benchmark.py
│   └── run_ablation_study.py
├── main.py                   # Entry point for the application
└── requirements.txt          # Python dependencies
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SerinaRica/EduRiskX.git
    cd EduRiskX
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset

This project is designed to work with the **Open University Learning Analytics Dataset (OULAD)**.
1.  Download the dataset from [Kaggle](https://www.kaggle.com/anlgrbz/student-demographics-online-education-dataoulad) or the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/open+university+learning+analytics+dataset).
2.  Place the CSV files (e.g., `studentInfo.csv`, `studentVle.csv`, etc.) in the `data/raw/` directory.

## 🏃 Usage

### Running Benchmarks
To run a comprehensive benchmark comparing different models:

```bash
python main.py benchmark
```

### Running Analysis
To perform a deep analysis of specific predictors and F-Logic reasoning:

```bash
python main.py analysis
```

## 🧠 How It Works

1.  **Data Preprocessing**: Raw OULAD data is cleaned, and temporal features (weekly interactions) are extracted.
2.  **Neural Prediction**: A Transformer-based model processes the sequence of student activities to predict the probability of failure/withdrawal.
3.  **Symbolic Reasoning**:
    *   The system evaluates the student's current state against a set of F-Logic rules (e.g., "If interaction decreases significantly AND assessment score is low, THEN risk is High").
    *   These rules are aligned with educational theories.
4.  **Evidence Aggregation**: The neural probability and symbolic belief are aggregated to form a final, robust risk assessment.
5.  **Output**: A risk report containing the probability, severity, explanations, and suggested interventions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
