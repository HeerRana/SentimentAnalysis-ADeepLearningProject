# Sentiment Analysis Using Deep Learning

This project performs binary sentiment classification on movie reviews using deep learning models built with TensorFlow and Keras. The goal is to predict whether a review is **positive** or **negative** using the IMDb movie review dataset.

## Project Overview

The project explores text preprocessing, vectorization, model training, evaluation, and prediction for sentiment analysis. Two notebook versions are included:

- `main.ipynb` contains the earlier workflow and an LSTM-based model.
- `main2.ipynb` contains a cleaner and improved pipeline with saved plots, confusion matrix, and model export.

## Dataset

The project uses the **IMDb ACL dataset** stored locally in the `aclImdb/` folder.

- Training data and testing data are loaded from directory structure
- Reviews are labeled as positive or negative
- Text data is converted into numeric form before being passed to the model

## Models Used

### 1. LSTM-based model (`main.ipynb`)

This notebook includes:

- Text vectorization with `TextVectorization`
- Embedding layer
- LSTM-based sentiment classifier
- A second deeper LSTM experiment with dropout
- Manual prediction on custom reviews
- Confusion matrix and classification report

One recorded test result from this notebook is approximately **78.46% accuracy**.

### 2. Deep learning text classifier (`main2.ipynb`)

This notebook includes:

- Dataset loading with `text_dataset_from_directory`
- Vocabulary building and text vectorization
- Embedding layer
- Global average pooling
- Dense classification layers
- Training and validation curves
- Confusion matrix and classification report
- Export and saving of the trained model

One recorded test result from this notebook is **84.00% accuracy**.

## Project Files

- `main.ipynb` - initial notebook with LSTM-based sentiment analysis
- `main2.ipynb` - improved notebook with cleaner training and evaluation pipeline
- `sentiment_model.keras` - saved trained model
- `training_curves.png` - plot of accuracy and loss over epochs
- `confusion_matrix.png` - confusion matrix generated on the test set
- `Deep Learning Project.pptx` - project presentation

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## How to Run

1. Create and activate a virtual environment.
2. Install the required packages.
3. Open the notebooks and run the cells in order.

Example package installation:

```bash
pip install tensorflow==2.15 numpy matplotlib seaborn scikit-learn jupyter
```

To start Jupyter:

```bash
jupyter notebook
```

## Output

The project produces:

- sentiment predictions for movie reviews
- model performance metrics
- training and validation plots
- confusion matrix for classification analysis
- a saved `.keras` model for reuse

## Learning Outcomes

This project demonstrates:

- text preprocessing for deep learning
- vectorization of natural language data
- training neural networks for NLP tasks
- comparing model performance
- evaluating classification models with more than just accuracy

## Future Improvements

- try Bidirectional LSTM or GRU models
- tune sequence length and vocabulary size
- add early stopping and model checkpointing
- convert notebook code into reusable Python scripts
- deploy the model with a simple web app

## Author

Deep Learning project on sentiment analysis using the IMDb dataset.
