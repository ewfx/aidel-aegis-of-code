# 🛡️ AegisofCode - AI-Driven Entity Risk Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)


## Overview
This project is built for the AI-Driven Risk Analysis Hackathon. It provides an automated solution to identify, enrich, classify, and risk-score entities (e.g., corporations, non-profits, shell companies) from structured and unstructured transaction data using Generative AI and open datasets.

## 🔍 Key Features
- Named Entity Recognition (NER) from transaction text
- Similarity search using FAISS and Sentence Transformers
- Heuristic risk scoring with fuzzy logic and keywords
- Justification generation using FLAN-T5 LLM
- Interactive UI using Gradio
- Output as structured JSON with confidence scores and reasons

## 📁 Repository Structure
```
/AegisofCode
├── artifacts/            # Sample outputs
├── arch/                 # Architecture documentation
├── code/                 # Main logic and Gradio UI
│   ├── entity_risk_analysis.py
│   └── gradio_app.py
├── demo/                 # Demo script
├── test/                 # Sample test cases
└── README.md             # Project instructions
```

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Install dependencies
```bash
pip install -r requirements.txt
```

You can also install manually:
```bash
pip install pandas faiss-cpu torch fuzzywuzzy sentence-transformers transformers gradio
```

## 🚀 How to Run

### 1. CLI Execution (for testing)
```bash
python code/entity_risk_analysis.py
```

### 2. Run Gradio UI
```bash
python code/gradio_app.py
```
Then visit `http://localhost:7860` in your browser.

## 🧪 Example Input
```
Payment made to XBridge Ventures, previously flagged by compliance.
```

## 🧠 Sample Output
```json
{
  "input_text": "...",
  "entities": [
    {
      "entity": "XBridge Ventures",
      "matched_to": "XBridge Ventures",
      "match_score": 95,
      "risk_score": 85,
      "justification": "This entity has been previously flagged in compliance reports..."
    }
  ]
}
```

## 📎 References
- [OpenCorporates API](https://api.opencorporates.com)
- [Wikidata](https://www.wikidata.org)
- [OFAC Sanctions List](https://www.treasury.gov/resource-center/sanctions)
- HuggingFace Transformers + Sentence Transformers

## 📽️ Demo Script
- Enter any suspicious or high-value transaction description.
- Click `Analyze` in the Gradio interface.
- View risk scores, classifications, and generated justifications.

## 🧠 Team: AegisofCode
Built with ❤️ for the Hackathon challenge.

