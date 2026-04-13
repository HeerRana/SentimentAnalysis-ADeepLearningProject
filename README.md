# 🎬 Sentiment Classification of Movie Reviews
### Using Bidirectional LSTM and Natural Language Processing

---

## 📌 Project Overview

This project builds a deep learning model that automatically classifies movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP) and a Bidirectional LSTM neural network.

- **Dataset** — IMDb Large Movie Review Dataset (50,000 reviews)
- **Framework** — TensorFlow / Keras (Python)
- **Model** — Bidirectional LSTM
- **Test Accuracy** — 83.52%

---

## 📁 Project Structure

```
project/
│
├── sentiment_analysis_imdb.ipynb     ← Main notebook (training + evaluation)
├── predict.py                        ← Run this for live predictions (no retraining)
│
├── sentiment_model.keras             ← Saved trained model (auto-generated)
├── vectorizer_vocab.pkl              ← Saved vocabulary (auto-generated)
│
├── training_curves.png               ← Accuracy & loss plots (auto-generated)
├── confusion_matrix.png              ← Confusion matrix (auto-generated)
├── model_architecture.png            ← Model layer diagram (auto-generated)
├── review_length_distribution.png    ← Review length histogram (auto-generated)
│
├── aclImdb/                          ← IMDb dataset folder
│   ├── train/
│   │   ├── pos/                      ← 12,500 positive reviews
│   │   └── neg/                      ← 12,500 negative reviews
│   └── test/
│       ├── pos/                      ← 12,500 positive reviews
│       └── neg/                      ← 12,500 negative reviews
│
└── README.md                         ← This file
```

> ⚠️ The `unsup/` folder inside `aclImdb/train/` must be deleted before running.
> It contains unlabeled data and creates a 3rd class which breaks binary classification.

---

## ⚙️ Setup Instructions

### Step 1 — Create virtual environment
```bash
python -m venv venv
```

### Step 2 — Activate virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install tensorflow notebook ipykernel matplotlib scikit-learn seaborn
```

### Step 4 — Download the dataset
Download the IMDb dataset from:
https://ai.stanford.edu/~amaas/data/sentiment/

Extract it so the folder structure looks like `aclImdb/train/pos/`, `aclImdb/train/neg/` etc.
Delete the `aclImdb/train/unsup/` folder before running.

---

## 🚀 How to Run

### Option A — Train the model (first time only)

Open and run all cells in:
```
sentiment_analysis_imdb.ipynb
```

This will:
- Load and preprocess the dataset
- Analyse review length distribution
- Build and train the Bidirectional LSTM model
- Evaluate on test set and print accuracy
- Save all plots as PNG files
- Save the trained model as `sentiment_model.keras`
- Save the vocabulary as `vectorizer_vocab.pkl`
- Launch interactive prediction at the end

**Expected training time:** 10-15 minutes (CPU)

---

### Option B — Run predictions only (no retraining)

Once the model is trained and saved, use this for live demos:

```bash
python predict.py
```

Loads in 3-4 seconds. Type any movie review and get instant results:

```
Enter a movie review (or type 'quit' to exit):
> This movie was absolutely fantastic!

Review    : This movie was absolutely fantastic!...
Sentiment : POSITIVE
Confidence: 94.3%
Raw Score : 0.9430
```

Type `quit` to exit.

---

## 🧠 Model Architecture

| Layer | Configuration | Purpose |
|---|---|---|
| Embedding | Vocab=20001, dim=32 | Word → dense vector |
| SpatialDropout1D | Rate=0.3 | Regularisation |
| Bidirectional LSTM | Units=64, return_seq=True | Forward + backward context |
| Bidirectional LSTM | Units=32 | Sequence summary |
| Dense | Units=64, ReLU | Feature learning |
| Dropout | Rate=0.5 | Regularisation |
| Dense (Output) | Units=1, Sigmoid | 0.0–1.0 probability |

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | 83.52% |
| Test Loss | 0.3965 |
| Positive F1-Score | 0.84 |
| Negative F1-Score | 0.83 |

---

## 🔧 Key Hyperparameters

| Parameter | Value |
|---|---|
| MAX_TOKENS | 20,000 |
| SEQUENCE_LEN | 250 |
| EMBEDDING_DIM | 32 |
| BATCH_SIZE | 32 |
| EPOCHS | 20 (with early stopping) |
| Early Stopping Patience | 5 |
| Optimizer | Adam (lr=0.001) |
| Loss Function | Binary Cross-Entropy |

---

## 📦 Requirements

- Python 3.11
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

---

## ⚠️ Known Limitations

- Model struggles with **double negatives** (e.g. "not bad" classified as negative)
- Reviews longer than 250 words are **truncated** (affects 30% of dataset)
- Trained only on movie reviews — may not generalise to other domains
- Sarcasm and mixed-sentiment reviews may be misclassified

---

## 🔮 Future Improvements

- Use pre-trained GloVe or FastText embeddings
- Fine-tune BERT for 93%+ accuracy
- Add negation preprocessing
- Deploy as a web app using FastAPI + HTML frontend

---

## 👨‍💻 Author

**Project:** Deep Learning — Semester 6
**Model:** Bidirectional LSTM Sentiment Classifier
**Tools:** Python, TensorFlow, Keras, IMDb Dataset