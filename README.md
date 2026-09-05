# NL2SQL Research — Practical Implementation

A Streamlit-based research platform for comparing 4 NL2SQL models:
**Rule-Based NLP**, **Random Forest**, **LSTM Seq2Seq**, and **T5 Transformer**.

---

## 🤗 Hosted Model

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-nl2sql--comparative--study--conit2026-FFD21E?style=for-the-badge)](https://huggingface.co/Ganesh-Nadkarni/nl2sql-comparative-study-conit2026)

This model accompanies the paper *"Comparative Study of AI Models for Natural Language to SQL Query Generation"* (CONIT 2026, IEEE Xplore).

---

## 📁 Project Structure

```
nl2sql_research/
├── app.py                      # Streamlit application (run this)
├── train.py                    # Training script (run this first)
├── requirements.txt
├── README.md
├── data/
│   ├── NL2SQL_Query_Dataset.csv  ← Place your dataset here
│   ├── preprocess.py
│   ├── split_data.py
│   └── splits/                 ← Auto-generated after training
│       ├── train.csv
│       ├── familiar_test.csv
│       └── unseen_test.csv
├── models/
│   ├── __init__.py
│   ├── rule_based.py
│   ├── random_forest_model.py
│   ├── lstm_model.py
│   └── t5_model.py
├── utils/
│   ├── __init__.py
│   ├── db_utils.py
│   └── evaluate.py
└── trained_models/             ← Auto-generated after training
    ├── rule_based.pkl
    ├── random_forest.pkl
    ├── lstm/
    ├── t5/
    └── all_metrics.json
```

---

## ⚙️ Setup

### 1. Prerequisites
- Python 3.9+
- 8GB+ RAM (16GB recommended for T5)
- GPU optional but recommended for LSTM/T5

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

For GPU support (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Place Dataset
Ensure `NL2SQL_Query_Dataset.csv` is in the `data/` folder:
```
data/NL2SQL_Query_Dataset.csv
```

---

## 🚂 Training

### Train All Models (recommended)
```bash
python train.py
```

### Train Specific Models
```bash
# Only fast models (Rule-Based + Random Forest)
python train.py --models rule rf

# Only deep learning models
python train.py --models lstm t5

# Quick test with fewer samples
python train.py --max-samples 500 --models rule rf
```

### Training Options
```
--models {rule,rf,lstm,t5,all}    Models to train (default: all)
--epochs-lstm INT                  LSTM epochs (default: 15)
--epochs-t5 INT                    T5 epochs (default: 5)
--batch-lstm INT                   LSTM batch size (default: 64)
--batch-t5 INT                     T5 batch size (default: 16)
--max-samples INT                  Limit training samples for quick test
```

### Expected Training Times (CPU)
| Model | Time |
|-------|------|
| Rule-Based NLP | ~30 seconds |
| Random Forest | ~3-8 minutes |
| LSTM Seq2Seq | ~15-45 minutes |
| T5 Transformer | ~1-3 hours |

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Opens at: http://localhost:8501

---

## 📱 App Features

### Tab 1 — Familiar Query
- Select from NL phrasings seen during training
- Run all models simultaneously
- Compare expected vs predicted SQL side-by-side
- Execute queries on demo SQLite database

### Tab 2 — Unseen Query
- Select NL phrasings the model has NEVER seen
- Model knows the SQL structure but not this exact wording
- Tests generalization ability

### Tab 3 — Run Familiar Test
- Batch test 10–200 familiar queries
- Accuracy bar charts (template + exact match)
- Per-model detailed results table

### Tab 4 — Run Unseen Test
- Batch test 10–200 unseen queries
- Cross-tab comparison showing accuracy drop from familiar → unseen

---

## 📊 Evaluation Metrics

- **Template Match Accuracy**: Matches SQL structural pattern (keywords + table names)
- **Exact Match Accuracy**: Normalized exact SQL string match

---

## 🗄️ Database

The app uses a local **SQLite** database (`nl2sql_demo.db`) with a schema
matching the dataset's tables. Sample data is auto-generated for demo purposes.

For a real setup with **MySQL Workbench**:
1. Import the schema: `mysql -u root -p < schema.sql`
2. Update `utils/db_utils.py` to use `pymysql` instead of `sqlite3`

---

## 📋 Data Split Strategy

```
Dataset (14,815 rows, 3,989 unique SQL queries)
    │
    ├── Training (80% of NL phrasings per SQL) → familiar_train
    │       └── Familiar Test ← subset of training phrasings (model HAS seen)
    │
    └── Unseen Test (20% of NL phrasings per SQL)
            └── Same SQL structures, different NL wording (model has NOT seen)
```
