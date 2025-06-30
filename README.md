
# 🧠 BERT-Based Multi-Domain Sentiment Analysis Framework

> **Aspect-Category-Opinion-Sentiment (ACOS) Extraction across E-commerce, Hospitality, and Online Education using BERT-mini**

![Model Diagram](./assets/model_architecture.png)

## 📌 Overview

This repository contains the official implementation of the paper:

**"A BERT-Based Framework for Multi-Domain Sentiment Analysis on Real-World Review Data"**

We propose a lightweight yet robust sentiment analysis pipeline using a fine-tuned BERT-mini model. It enables aspect-level sentiment extraction across multiple domains — including e-commerce, hospitality, and online education — with an emphasis on extracting **Aspect–Category–Opinion–Sentiment (ACOS)** structures.

---

## 🔍 Features

- **Multi-domain support** (Amazon, Hotels, Coursera)
- **Lightweight BERT-mini model** for faster inference
- **Aspect-based sentiment analysis** with full ACOS quadruple extraction
- Balanced training via **Stratified K-Fold Cross-Validation**
- Clear metrics: Accuracy, Precision, Recall, Macro/Weighted F1
- Error analysis included to improve domain interpretability

---

## Project Structure

```bash
├── data/
│   ├── raw/
│   ├── processed/
│   └── stats/
├── models/
│   └── checkpoints/
├── notebooks/
│   ├── training.ipynb
│   └── evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── assets/
│   └── model_architecture.png
├── requirements.txt
└── README.md
```

---

## Dataset

We use the **OATS Dataset** curated from real-world review platforms. Each review is labeled with:

- **Sentiment** (`positive`, `neutral`, `negative`)
- **Aspect**
- **Category**
- **Opinion**

| Domain     | Train | Val  | Test | Total  |
|------------|-------|------|------|--------|
| Amazon     | 7360  | 920  | 920  | 9200   |
| Coursera   | 7188  | 898  | 899  | 8985   |
| Hotels     | 7834  | 955  | 980  | 9769   |
| **Total**  | -     | -    | -    | 27,954 |

---

## Model Details

We use a compact version of BERT:

- **Model**: [`prajjwal1/bert-mini`](https://huggingface.co/prajjwal1/bert-mini)
- **Parameters**: 4-layer, 4-head attention, 256 hidden size
- **Fine-tuning Task**: 3-class sentiment classification (`pos`, `neu`, `neg`)

### Training Config

| Parameter         | Value            |
|------------------|------------------|
| Batch size        | 512              |
| Learning rate     | 1e-5             |
| Max epochs        | 50               |
| Early Stopping    | 30 epochs        |
| Optimizer         | AdamW            |
| Loss Function     | CrossEntropyLoss |
| CV folds (k)      | 20               |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/bert-multidomain-sentiment.git
cd bert-multidomain-sentiment
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/train.py --dataset data/processed/k_fold_merge_train_valid.csv --model bert-mini
```

### 4. Evaluate

```bash
python src/evaluate.py --model checkpoints/best_model.pt --test data/processed/test.csv
```

---

## Results

| Domain     | Accuracy | Macro-F1 | Neutral F1 |
|------------|----------|----------|-------------|
| Amazon     | 88.4%    | 85.6%    | 24%         |
| Coursera   | 91.1%    | 89.7%    | 28%         |
| Hotels     | 89.2%    | 86.5%    | 26%         |
| **Combined**| **92.3%**| **90.2%**| **27%**     |

---

## Error Analysis

Common error types:
- Missing `aspect` or `opinion` in valid `sentiment` entries
- Ambiguous expressions like _"Love it!"_ or _"Highly recommended."_
- Incomplete phrases (e.g., _"They taste like"_)

---

## 🔮 Future Work

- Integrate domain-adaptive pretraining
- Use prompt-tuning for low-resource classes
- Enhance with external knowledge graphs
- Extend to other domains (healthcare, finance)

---

## Citation

```bibtex
@article{huynh2025sentiment,
  title={A BERT-Based Framework for Multi-Domain Sentiment Analysis on Real-World Review Data},
  author={Huynh, Phuong Vi and Tran, Tuyet Hue and Nguyen-Thi, Cam-Tien},
  journal={Preprint},
  year={2025}
}
```

---

## Acknowledgments

- OATS Dataset by Chebolu et al. (2024)
- HuggingFace Transformers
- PyTorch & scikit-learn

---

## 📬 Contact

For questions, reach out to:
- **Tuyet Hue Tran**: [23C15027@student.hcmus.edu.vn](mailto:23C15027@student.hcmus.edu.vn)
